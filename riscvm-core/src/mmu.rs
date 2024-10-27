#[derive(Debug)]
pub struct Mmu {}

#[derive(Debug, Clone, Copy)]
pub struct PageTableEntry {
    entry: u64,
}

impl PageTableEntry {
    // Constants for flag bits
    const FLAG_V: u64 = 1 << 0;
    const FLAG_R: u64 = 1 << 1;
    const FLAG_W: u64 = 1 << 2;
    const FLAG_X: u64 = 1 << 3;
    const FLAG_U: u64 = 1 << 4;
    const FLAG_G: u64 = 1 << 5;
    const FLAG_A: u64 = 1 << 6;
    const FLAG_D: u64 = 1 << 7;

    // Extract PPN (Physical Page Number)
    fn ppn(&self) -> u64 {
        (self.entry >> 10) & 0xFFFFFFF
    }

    // Check if the entry is valid
    fn is_valid(&self) -> bool {
        self.entry & Self::FLAG_V != 0
    }

    // Check if the entry is a leaf (points to a page, not a table)
    fn is_leaf(&self) -> bool {
        self.entry & (Self::FLAG_R | Self::FLAG_X | Self::FLAG_W) != 0
    }

    // Get flags
    fn flags(&self) -> u64 {
        self.entry & 0x3FF // Bits 9-0
    }
}
