use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::optimizer::OptimizationReport;
use super::{prepare_plan_for_tier, BlockPlan, CompiledBlock, JitError, JitTier, NativeBackend};

pub(super) struct BackgroundCompiler {
    task_tx: SyncSender<BackgroundCompileJob>,
    result_rx: Receiver<BackgroundCompileResult>,
    workers: Vec<JoinHandle<()>>,
}

impl BackgroundCompiler {
    pub(super) fn new(worker_count: usize, queue_limit: usize) -> Result<Self, JitError> {
        let (task_tx, task_rx) = mpsc::sync_channel(queue_limit.max(1));
        let (result_tx, result_rx) = mpsc::channel();
        let task_rx = Arc::new(Mutex::new(task_rx));
        let mut workers: Vec<JoinHandle<()>> = Vec::with_capacity(worker_count.max(1));

        for index in 0..worker_count.max(1) {
            let task_rx = Arc::clone(&task_rx);
            let result_tx = result_tx.clone();
            let name = format!("riscvm-jit-compiler-{index}");
            let worker = match thread::Builder::new()
                .name(name)
                .spawn(move || worker_loop(task_rx, result_tx))
            {
                Ok(worker) => worker,
                Err(error) => {
                    drop(task_tx);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(JitError::BackgroundThreadSpawnFailed {
                        reason: error.to_string(),
                    });
                }
            };
            workers.push(worker);
        }

        Ok(Self {
            task_tx,
            result_rx,
            workers,
        })
    }

    pub(super) fn try_enqueue(
        &self,
        job: BackgroundCompileJob,
    ) -> Result<(), BackgroundEnqueueError> {
        self.task_tx.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => BackgroundEnqueueError::Full,
            TrySendError::Disconnected(_) => BackgroundEnqueueError::Disconnected,
        })
    }

    pub(super) fn drain_ready(&mut self) -> Vec<BackgroundCompileResult> {
        let mut results = Vec::new();
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => results.push(result),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        results
    }
}

impl Drop for BackgroundCompiler {
    fn drop(&mut self) {
        let (replacement_tx, _replacement_rx) = mpsc::sync_channel(1);
        let task_tx = std::mem::replace(&mut self.task_tx, replacement_tx);
        drop(task_tx);

        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub(super) enum BackgroundEnqueueError {
    Full,
    Disconnected,
}

pub(super) struct BackgroundCompileJob {
    pub pc: u64,
    pub plan: BlockPlan,
    pub tier: JitTier,
    pub reason: &'static str,
    pub preserved_execution_count: u64,
    pub include_listing: bool,
}

pub(super) enum BackgroundCompileResult {
    Compiled {
        pc: u64,
        tier: JitTier,
        reason: &'static str,
        preserved_execution_count: u64,
        plan: BlockPlan,
        optimization_report: OptimizationReport,
        block: CompiledBlock,
        compile_duration: Duration,
    },
    PromotedWithoutCompile {
        pc: u64,
        tier: JitTier,
        reason: &'static str,
        optimization_report: OptimizationReport,
    },
    Failed {
        pc: u64,
        tier: JitTier,
        reason: &'static str,
        error: JitError,
    },
    WorkerFailed {
        error: JitError,
    },
}

fn worker_loop(
    task_rx: Arc<Mutex<Receiver<BackgroundCompileJob>>>,
    result_tx: mpsc::Sender<BackgroundCompileResult>,
) {
    let mut backend = NativeBackend::new();

    loop {
        let job = {
            let Ok(task_rx) = task_rx.lock() else {
                let _ = result_tx.send(BackgroundCompileResult::WorkerFailed {
                    error: JitError::BackgroundCompilerPoisoned,
                });
                break;
            };
            task_rx.recv()
        };

        let Ok(job) = job else {
            break;
        };

        if result_tx.send(compile_job(&mut backend, job)).is_err() {
            break;
        }
    }
}

fn compile_job(backend: &mut NativeBackend, job: BackgroundCompileJob) -> BackgroundCompileResult {
    let BackgroundCompileJob {
        pc,
        plan,
        tier,
        reason,
        preserved_execution_count,
        include_listing,
    } = job;

    let prepared = prepare_plan_for_tier(plan, tier);
    if prepared.skipped_compile {
        return BackgroundCompileResult::PromotedWithoutCompile {
            pc,
            tier,
            reason,
            optimization_report: prepared.optimization_report,
        };
    }

    let compile_start = Instant::now();
    let plan = prepared.plan;
    match backend.compile(&plan, tier, include_listing) {
        Ok(mut block) => {
            block.tier = tier;
            block.execution_count = preserved_execution_count;
            BackgroundCompileResult::Compiled {
                pc,
                tier,
                reason,
                preserved_execution_count,
                plan,
                optimization_report: prepared.optimization_report,
                block,
                compile_duration: compile_start.elapsed(),
            }
        }
        Err(error) => BackgroundCompileResult::Failed {
            pc,
            tier,
            reason,
            error,
        },
    }
}
