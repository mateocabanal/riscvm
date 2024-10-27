use thiserror::Error;

#[derive(Debug, Error)]
pub enum Exception {
    #[error("Illegal Instruction")]
    IllegalInstruction,
}
