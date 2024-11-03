use std::fmt::Display;

use tracing::trace;

#[derive(Debug)]
pub struct Ram {
    regions: Vec<MemoryRegion>,
    pub lowest_unalloced_addr: u64,
}

impl Default for Ram {
    fn default() -> Self {
        Self::new()
    }
}

impl Ram {
    pub fn new() -> Ram {
        Ram {
            regions: Vec::new(),
            lowest_unalloced_addr: 0,
        }
    }

    pub fn add_region(&mut self, region: MemoryRegion) -> Result<(), MemoryError> {
        // Check for overlaps
        if let Some(overlap) = self.find_overlap(&region) {
            return Err(MemoryError::RegionOverlap(overlap.start));
        }

        // Find the insertion index
        let index = self
            .regions
            .binary_search_by_key(&region.start, |r| r.start)
            .unwrap_or_else(|e| e);

        if region.start + region.size > self.lowest_unalloced_addr {
            self.lowest_unalloced_addr = region.start + region.size;
        }

        // Insert the region at the correct position
        self.regions.insert(index, region);

        Ok(())
    }

    pub fn extend_region(&mut self, addr: u64, addition: u64) -> Result<(), MemoryError> {
        let region = self
            .find_region_mut(addr)
            .ok_or(MemoryError::InvalidAddress(addr));

        let Ok(region) = region else {
            return region.map(|i| ());
        };

        region.extend(addition);

        Ok(())
    }

    pub fn extend_text_region(&mut self, addition: u64) -> Result<(), MemoryError> {
        let region = self.regions.iter_mut().find(|reg| reg.is_text()).unwrap();
        region.extend(addition);

        Ok(())
    }

    pub fn remove_region(&mut self, addr: u64) -> Result<(), MemoryError> {
        let region = self
            .find_region(addr)
            .ok_or(MemoryError::InvalidAddress(addr));

        let Ok(region) = region else {
            return region.map(|i| ());
        };

        let index = self
            .regions
            .binary_search_by_key(&region.start, |r| r.start)
            .unwrap();

        if region.start + region.size == self.lowest_unalloced_addr {
            self.lowest_unalloced_addr = self.regions.last().map(|i| i.start + i.size).unwrap_or(0);
        }

        self.regions.remove(index);

        Ok(())
    }

    fn find_overlap(&self, new_region: &MemoryRegion) -> Option<&MemoryRegion> {
        // Check for overlap with existing regions
        self.regions
            .iter()
            .find(|&region| Ram::regions_overlap(region, new_region))
    }

    fn find_region(&self, address: u64) -> Option<&MemoryRegion> {
        let mut low = 0;
        let mut high = self.regions.len();

        while low < high {
            let mid = (low + high) / 2;
            let region = &self.regions[mid];
            if address < region.start {
                high = mid;
            } else if address >= region.start + region.size {
                low = mid + 1;
            } else {
                return Some(region);
            }
        }
        None
    }

    fn find_region_mut(&mut self, address: u64) -> Option<&mut MemoryRegion> {
        let mut low = 0;
        let mut high = self.regions.len();

        while low < high {
            let mid = (low + high) / 2;
            let region_start = self.regions[mid].start;
            let region_end = region_start + self.regions[mid].size;
            if address < region_start {
                high = mid;
            } else if address >= region_end {
                low = mid + 1;
            } else {
                return Some(&mut self.regions[mid]);
            }
        }
        None
    }

    pub fn read_byte(&self, address: u64) -> Result<u8, MemoryError> {
        let region = self
            .find_region(address)
            .ok_or(MemoryError::InvalidAddress(address))?;
        let offset = (address - region.start) as usize;
        Ok(region.data[offset])
    }

    pub fn write_byte(&mut self, address: u64, value: u8) -> Result<(), MemoryError> {
        let region = self
            .find_region_mut(address)
            .ok_or(MemoryError::InvalidAddress(address))?;
        trace!("addr: {address:08x}");
        let offset = (address - region.start) as usize;
        region.data[offset] = value;
        Ok(())
    }

    pub fn read_halfword(&self, address: u64) -> Result<u64, MemoryError> {
        self.read_nbytes(address, 2)
    }

    pub fn write_halfword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        self.write_nbytes(address, value, 2)
    }

    pub fn read_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        self.read_nbytes(address, 8)
    }

    pub fn write_doubleword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        self.write_nbytes(address, value, 8)
    }

    pub fn read_nbytes(&self, address: u64, len: u64) -> Result<u64, MemoryError> {
        let mut result = Vec::new();
        for addr in address..address + len {
            result.push(self.read_byte(addr)?)
        }

        Ok(result
            .iter()
            .enumerate()
            .fold(0u64, |res, (idx, val)| res + (u64::from(*val) << (8 * idx))))
    }

    pub fn write_nbytes(&mut self, address: u64, value: u64, len: u64) -> Result<(), MemoryError> {
        for (idx, addr) in (address..address + len).enumerate() {
            self.write_byte(addr, ((value >> (8 * idx)) & 0xFF) as u8)?;
        }

        Ok(())
    }

    pub fn read_word(&self, address: u64) -> Result<u32, MemoryError> {
        let b0 = self.read_byte(address)? as u32;
        let b1 = self.read_byte(address + 1)? as u32;
        let b2 = self.read_byte(address + 2)? as u32;
        let b3 = self.read_byte(address + 3)? as u32;
        Ok(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    pub fn write_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        self.write_byte(address, (value & 0xFF) as u8)?;
        self.write_byte(address + 1, ((value >> 8) & 0xFF) as u8)?;
        self.write_byte(address + 2, ((value >> 16) & 0xFF) as u8)?;
        self.write_byte(address + 3, ((value >> 24) & 0xFF) as u8)?;
        Ok(())
    }

    #[inline]
    fn regions_overlap(a: &MemoryRegion, b: &MemoryRegion) -> bool {
        let a_end = a.start + a.size;
        let b_end = b.start + b.size;
        a.start < b_end && b.start < a_end
    }
}

impl Display for Ram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for region in self.regions.iter() {
            writeln!(f, "{region}")?
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct MemoryRegion {
    start: u64,
    size: u64,
    flags: u64,
    data: Vec<u8>,
}

impl Display for MemoryRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start: {:08x}, size: {}, data: ...",
            self.start, self.size
        )
    }
}

impl MemoryRegion {
    pub fn new(start: u64, size: u64, data: Vec<u8>) -> Self {
        MemoryRegion {
            start,
            size,
            data,
            flags: 0,
        }
    }

    pub fn new_with_flags(start: u64, size: u64, data: Vec<u8>, flags: u64) -> Self {
        MemoryRegion {
            start,
            size,
            data,
            flags,
        }
    }

    pub fn is_text(&self) -> bool {
        self.flags & 1 == 1
    }

    pub fn extend(&mut self, addition: u64) {
        self.data.extend(vec![0u8; addition as usize]);
        self.size += addition;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Invalid address: {0:08x}")]
    InvalidAddress(u64),
    #[error("Permission denied at address: 0x{0:X}")]
    PermissionDenied(u64),
    #[error("Region overlap at address: 0x{0:X}")]
    RegionOverlap(u64),
}
