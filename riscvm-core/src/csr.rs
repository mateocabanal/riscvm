use crate::exception::Exception;

const CSR_SSTATUS: u64 = 0x100;
const CSR_SIE: u64 = 0x104;
const CSR_STVEC: u64 = 0x105;
const CSR_SSCRATCH: u64 = 0x140;
const CSR_SEPC: u64 = 0x141;
const CSR_SCAUSE: u64 = 0x142;
const CSR_STVAL: u64 = 0x143;
const CSR_SATP: u64 = 0x180;

pub struct Csr {
    satp: u64,
    sstatus: u64,
    stvec: u64,
    sepc: u64,
    scause: u64,
    stval: u64,
    sscratch: u64,
}

impl Csr {
    pub fn new() -> Csr {
        Csr {
            satp: 0,
            sstatus: 0,
            stvec: 0,
            sepc: 0,
            scause: 0,
            stval: 0,
            sscratch: 0,
        }
    }

    pub fn read(&self, addr: u64) -> u64 {
        match addr {
            CSR_SSTATUS => self.sstatus,
            CSR_STVEC => self.stvec,
            CSR_SSCRATCH => self.sscratch,
            CSR_SEPC => self.sepc,
            CSR_SCAUSE => self.scause,
            CSR_STVAL => self.stval,
            CSR_SATP => self.satp,

            _ => todo!(),
        }
    }

    pub fn write(&mut self, addr: u64, value: u64) -> Result<(), Exception> {
        match addr {
            CSR_SSTATUS => {
                todo!();
                // Handle writeable bits; preserve read-only bits
                // let mask = /* mask of writeable bits */
                // self.sstatus = (self.sstatus & !mask) | (value & mask);
                // Ok(())
            }
            CSR_STVEC => {
                self.stvec = value;
                Ok(())
            }
            CSR_SSCRATCH => {
                self.sscratch = value;
                Ok(())
            }
            CSR_SEPC => {
                self.sepc = value;
                Ok(())
            }
            CSR_SCAUSE => {
                self.scause = value;
                Ok(())
            }
            CSR_STVAL => {
                self.stval = value;
                Ok(())
            }
            CSR_SATP => {
                self.satp = value;
                Ok(())
            }
            // ... handle other CSRs
            _ => Err(Exception::IllegalInstruction),
        }
    }
}
