use std::fmt::Display;

use thiserror::Error;
pub const PAGE_SIZE: u64 = 4096;
const MMAP_BASE: u64 = 0x4000_0000;
pub(crate) const JIT_FAST_CACHE_ENTRIES: usize = 8;

pub fn align_up(addr: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (addr + align - 1) & !(align - 1)
}

#[derive(Debug)]
pub struct Ram {
    regions: Vec<MemoryRegion>,
    cached_region_index: Option<usize>,
    cached_region_indices: [Option<usize>; JIT_FAST_CACHE_ENTRIES],
    jit_fast_caches: [JitFastMemoryCache; JIT_FAST_CACHE_ENTRIES],
    pub lowest_unalloced_addr: u64,
    program_break: u64,
    heap_start: u64,
    next_mmap_addr: u64,
    code_version: u64,
}

impl Clone for Ram {
    fn clone(&self) -> Self {
        Self {
            regions: self.regions.clone(),
            cached_region_index: None,
            cached_region_indices: [None; JIT_FAST_CACHE_ENTRIES],
            jit_fast_caches: [JitFastMemoryCache::default(); JIT_FAST_CACHE_ENTRIES],
            lowest_unalloced_addr: self.lowest_unalloced_addr,
            program_break: self.program_break,
            heap_start: self.heap_start,
            next_mmap_addr: self.next_mmap_addr,
            code_version: self.code_version,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct JitFastMemoryCache {
    pub start: u64,
    pub end: u64,
    pub read_ptr: *const u8,
    pub write_ptr: *mut u8,
}

impl Default for JitFastMemoryCache {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            read_ptr: std::ptr::null(),
            write_ptr: std::ptr::null_mut(),
        }
    }
}

pub(crate) const RAM_JIT_FAST_CACHES_OFFSET: usize = std::mem::offset_of!(Ram, jit_fast_caches);
pub(crate) const JIT_FAST_CACHE_START_OFFSET: usize =
    std::mem::offset_of!(JitFastMemoryCache, start);
pub(crate) const JIT_FAST_CACHE_END_OFFSET: usize = std::mem::offset_of!(JitFastMemoryCache, end);
pub(crate) const JIT_FAST_CACHE_READ_PTR_OFFSET: usize =
    std::mem::offset_of!(JitFastMemoryCache, read_ptr);
pub(crate) const JIT_FAST_CACHE_WRITE_PTR_OFFSET: usize =
    std::mem::offset_of!(JitFastMemoryCache, write_ptr);

impl Default for Ram {
    fn default() -> Self {
        Self::new()
    }
}

impl Ram {
    pub fn new() -> Ram {
        Ram {
            regions: Vec::new(),
            cached_region_index: None,
            cached_region_indices: [None; JIT_FAST_CACHE_ENTRIES],
            jit_fast_caches: [JitFastMemoryCache::default(); JIT_FAST_CACHE_ENTRIES],
            lowest_unalloced_addr: 0,
            program_break: 0,
            heap_start: 0,
            next_mmap_addr: MMAP_BASE,
            code_version: 0,
        }
    }

    pub fn code_version(&self) -> u64 {
        self.code_version
    }

    pub fn executable_ranges(&self) -> Vec<(u64, u64)> {
        self.regions
            .iter()
            .filter(|region| region.is_execute())
            .map(|region| (region.start, region.end()))
            .collect()
    }

    pub fn add_region(&mut self, region: MemoryRegion) -> Result<(), MemoryError> {
        let changes_code = region.is_execute();

        // Check for overlaps
        if let Some(overlap) = self.find_overlap(&region) {
            return Err(MemoryError::RegionOverlap(overlap.start));
        }

        // Find the insertion index
        let index = self
            .regions
            .binary_search_by_key(&region.start, |r| r.start)
            .unwrap_or_else(|e| e);

        if region.end() > self.lowest_unalloced_addr {
            self.lowest_unalloced_addr = region.end();
        }

        // Insert the region at the correct position
        self.regions.insert(index, region);
        self.invalidate_fast_cache();
        if changes_code {
            self.bump_code_version();
        }

        Ok(())
    }

    pub fn copy_data_from(&mut self, other: &Ram) {
        for other_region in &other.regions {
            if let Some(region) = self
                .regions
                .iter_mut()
                .find(|region| region.start == other_region.start)
            {
                region.size = other_region.size;
                region.flags = other_region.flags;
                region.data.clone_from(&other_region.data);
            } else {
                self.regions.push(other_region.clone());
            }
        }
        self.lowest_unalloced_addr = other.lowest_unalloced_addr;
        self.program_break = other.program_break;
        self.heap_start = other.heap_start;
        self.next_mmap_addr = other.next_mmap_addr;
        self.code_version = other.code_version;
        self.regions.sort_by_key(|region| region.start);
        self.invalidate_fast_cache();
    }

    pub fn extend_region(&mut self, addr: u64, addition: u64) -> Result<(), MemoryError> {
        let region = self
            .find_region_mut(addr)
            .ok_or(MemoryError::InvalidAddress(addr));

        let Ok(region) = region else {
            return region.map(|_| ());
        };

        let new_end = {
            region.extend(addition);
            region.end()
        };
        self.invalidate_fast_cache();
        if new_end > self.lowest_unalloced_addr {
            self.lowest_unalloced_addr = new_end;
        }

        Ok(())
    }

    pub fn set_program_break_base(&mut self, addr: u64) {
        let addr = align_up(addr, PAGE_SIZE);
        self.heap_start = addr;
        self.program_break = addr;
        if self.next_mmap_addr <= addr {
            self.next_mmap_addr = align_up(addr + PAGE_SIZE, PAGE_SIZE);
        }
    }

    pub fn program_break(&self) -> u64 {
        self.program_break
    }

    pub fn set_program_break(&mut self, addr: u64) -> Result<u64, MemoryError> {
        if addr < self.heap_start {
            return Err(MemoryError::InvalidAddress(addr));
        }

        if addr <= self.program_break {
            self.program_break = addr;
            return Ok(addr);
        }

        if let Some(idx) = self.region_index_starting_at(self.heap_start) {
            let current_end = self.regions[idx].end();
            if addr > current_end {
                if let Some(next_region) = self.regions.get(idx + 1) {
                    if next_region.start < addr {
                        return Err(MemoryError::RegionOverlap(next_region.start));
                    }
                }

                self.regions[idx].extend(addr - current_end);
                self.invalidate_fast_cache();
                if self.regions[idx].end() > self.lowest_unalloced_addr {
                    self.lowest_unalloced_addr = self.regions[idx].end();
                }
            }
        } else if addr > self.heap_start {
            self.add_region(MemoryRegion::new(
                self.heap_start,
                addr - self.heap_start,
                vec![0; (addr - self.heap_start) as usize],
            ))?;
        }

        self.program_break = addr;
        Ok(addr)
    }

    pub fn mmap_anonymous(
        &mut self,
        requested_addr: Option<u64>,
        len: u64,
    ) -> Result<u64, MemoryError> {
        self.mmap_anonymous_with_flags(requested_addr, len, 0, false)
    }

    pub fn mmap_anonymous_with_flags(
        &mut self,
        requested_addr: Option<u64>,
        len: u64,
        flags: u64,
        replace: bool,
    ) -> Result<u64, MemoryError> {
        let len = align_up(len, PAGE_SIZE);
        let addr = requested_addr
            .map(|addr| align_up(addr, PAGE_SIZE))
            .unwrap_or_else(|| self.next_free_mmap_addr(len));

        if replace {
            self.unmap_overlaps(addr, len)?;
        }

        self.add_region(MemoryRegion::new_with_flags(
            addr,
            len,
            vec![0; len as usize],
            flags,
        ))?;

        if requested_addr.is_none() {
            self.next_mmap_addr = align_up(addr + len, PAGE_SIZE);
        }

        Ok(addr)
    }

    pub fn mmap_bytes(
        &mut self,
        requested_addr: Option<u64>,
        len: u64,
        mut data: Vec<u8>,
        flags: u64,
        replace: bool,
    ) -> Result<u64, MemoryError> {
        let len = align_up(len, PAGE_SIZE);
        data.resize(len as usize, 0);
        let addr = requested_addr
            .map(|addr| align_up(addr, PAGE_SIZE))
            .unwrap_or_else(|| self.next_free_mmap_addr(len));

        if replace {
            self.unmap_overlaps(addr, len)?;
        }

        self.add_region(MemoryRegion::new_with_flags(addr, len, data, flags))?;

        if requested_addr.is_none() {
            self.next_mmap_addr = align_up(addr + len, PAGE_SIZE);
        }

        Ok(addr)
    }

    pub fn find_end_of_text_region(&self) -> u64 {
        let mut iter = self.regions.iter();

        // There has to be at least 1 region
        let mut biggest_reg = iter.next().unwrap();
        for reg in iter {
            if !reg.is_execute() {
                continue;
            }
            if reg.is_execute() && !biggest_reg.is_execute() {
                biggest_reg = reg;
                continue;
            }

            if reg.start > biggest_reg.start {
                biggest_reg = reg;
            }
        }

        biggest_reg.start + biggest_reg.size
    }

    pub fn extend_text_region_to(&mut self, addr: u64) -> Result<(), MemoryError> {
        let mut iter = self.regions.iter_mut();

        // There has to be at least 1 region
        let mut biggest_reg = iter.next().unwrap();
        for reg in iter {
            if !reg.is_execute() {
                continue;
            }
            if reg.is_execute() && !biggest_reg.is_execute() {
                biggest_reg = reg;
                continue;
            }

            if reg.start > biggest_reg.start {
                biggest_reg = reg;
            }
        }

        let Some(offset) = addr.checked_sub(biggest_reg.start + biggest_reg.size) else {
            return Err(MemoryError::InvalidAddress(u64::MAX));
        };

        let changes_code = biggest_reg.is_execute();
        biggest_reg.extend(offset);
        self.invalidate_fast_cache();
        if changes_code {
            self.bump_code_version();
        }

        Ok(())
    }

    pub fn remove_region(&mut self, addr: u64) -> Result<(), MemoryError> {
        let region = self
            .find_region(addr)
            .ok_or(MemoryError::InvalidAddress(addr));

        let Ok(region) = region else {
            return region.map(|_| ());
        };

        let index = self
            .regions
            .binary_search_by_key(&region.start, |r| r.start)
            .unwrap();
        let changes_code = region.is_execute();

        if region.start + region.size == self.lowest_unalloced_addr {
            self.lowest_unalloced_addr = self.regions.last().map(|i| i.start + i.size).unwrap_or(0);
        }

        self.regions.remove(index);
        self.invalidate_fast_cache();
        self.recompute_lowest_unalloced_addr();
        if changes_code {
            self.bump_code_version();
        }

        Ok(())
    }

    pub fn munmap(&mut self, addr: u64, len: u64) -> Result<(), MemoryError> {
        if len == 0 {
            return Err(MemoryError::InvalidAddress(addr));
        }

        let len = align_up(len, PAGE_SIZE);
        let end = addr
            .checked_add(len)
            .ok_or(MemoryError::InvalidAddress(addr))?;
        let index = self
            .regions
            .iter()
            .position(|region| addr >= region.start && end <= region.end())
            .ok_or(MemoryError::InvalidAddress(addr))?;

        let region_start = self.regions[index].start;
        let region_end = self.regions[index].end();
        let changes_code = self.regions[index].is_execute();
        if addr == region_start && end == region_end {
            self.regions.remove(index);
        } else if addr == region_start {
            let remove_len = (end - region_start) as usize;
            let region = &mut self.regions[index];
            region.data.drain(..remove_len);
            region.start = end;
            region.size = region_end - end;
        } else if end == region_end {
            let keep_len = (addr - region_start) as usize;
            let region = &mut self.regions[index];
            region.data.truncate(keep_len);
            region.size = addr - region_start;
        } else {
            let right_data = {
                let region = &mut self.regions[index];
                let split_at = (end - region_start) as usize;
                let right_data = region.data.split_off(split_at);
                region.data.truncate((addr - region_start) as usize);
                region.size = addr - region_start;
                right_data
            };
            let flags = self.regions[index].flags;
            self.regions.insert(
                index + 1,
                MemoryRegion::new_with_flags(end, region_end - end, right_data, flags),
            );
        }

        self.invalidate_fast_cache();
        self.recompute_lowest_unalloced_addr();
        if changes_code {
            self.bump_code_version();
        }

        Ok(())
    }

    fn find_overlap(&self, new_region: &MemoryRegion) -> Option<&MemoryRegion> {
        // Check for overlap with existing regions
        self.regions
            .iter()
            .find(|&region| Ram::regions_overlap(region, new_region))
    }

    fn region_index_starting_at(&self, start: u64) -> Option<usize> {
        self.regions.binary_search_by_key(&start, |r| r.start).ok()
    }

    fn next_free_mmap_addr(&self, len: u64) -> u64 {
        let mut candidate = self.next_mmap_addr;
        loop {
            let end = candidate + len;
            if let Some(overlap) = self
                .regions
                .iter()
                .find(|region| candidate < region.end() && region.start < end)
            {
                candidate = align_up(overlap.end(), PAGE_SIZE);
            } else {
                return candidate;
            }
        }
    }

    fn unmap_overlaps(&mut self, addr: u64, len: u64) -> Result<(), MemoryError> {
        let end = addr
            .checked_add(len)
            .ok_or(MemoryError::InvalidAddress(addr))?;
        let overlaps = self
            .regions
            .iter()
            .filter_map(|region| {
                let overlap_start = addr.max(region.start);
                let overlap_end = end.min(region.end());
                (overlap_start < overlap_end).then_some((overlap_start, overlap_end))
            })
            .collect::<Vec<_>>();

        for (overlap_start, overlap_end) in overlaps {
            self.munmap(overlap_start, overlap_end - overlap_start)?;
        }

        Ok(())
    }

    fn recompute_lowest_unalloced_addr(&mut self) {
        self.lowest_unalloced_addr = self
            .regions
            .iter()
            .map(MemoryRegion::end)
            .max()
            .unwrap_or(0);
    }

    fn find_region(&self, address: u64) -> Option<&MemoryRegion> {
        self.find_region_index(address)
            .map(|index| &self.regions[index])
    }

    fn find_region_index(&self, address: u64) -> Option<usize> {
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
                return Some(mid);
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

    fn find_region_range(&self, address: u64, len: usize) -> Option<&MemoryRegion> {
        let end = address.checked_add(len as u64)?;
        self.find_region(address)
            .filter(|region| end <= region.end())
    }

    fn find_region_range_mut(&mut self, address: u64, len: usize) -> Option<&mut MemoryRegion> {
        let end = address.checked_add(len as u64)?;
        self.find_region_mut(address)
            .filter(|region| end <= region.end())
    }

    fn cached_region_index_for_range(&mut self, address: u64, len: usize) -> Option<usize> {
        let end = address.checked_add(len as u64)?;

        if let Some(index) = self.cached_region_index {
            if let Some(region) = self.regions.get(index) {
                if address >= region.start && end <= region.end() {
                    return Some(index);
                }
            }
        }

        for index in self.cached_region_indices.into_iter().flatten() {
            if Some(index) == self.cached_region_index {
                continue;
            }
            if let Some(region) = self.regions.get(index) {
                if address >= region.start && end <= region.end() {
                    self.remember_region_index(index);
                    return Some(index);
                }
            }
        }

        let index = self.find_region_index(address)?;
        let region = &self.regions[index];
        if end <= region.end() {
            self.remember_region_index(index);
            Some(index)
        } else {
            None
        }
    }

    fn fast_cached_read_ptr_range(&self, address: u64, len: usize) -> Option<*const u8> {
        let end = address.checked_add(len as u64)?;
        for cache in &self.jit_fast_caches {
            if cache.read_ptr.is_null() {
                continue;
            }
            if address >= cache.start && end <= cache.end {
                let offset = (address - cache.start) as usize;
                return Some(cache.read_ptr.wrapping_add(offset));
            }
        }
        None
    }

    fn fast_cached_write_ptr_range(&self, address: u64, len: usize) -> Option<*mut u8> {
        let end = address.checked_add(len as u64)?;
        for cache in &self.jit_fast_caches {
            if cache.write_ptr.is_null() {
                continue;
            }
            if address >= cache.start && end <= cache.end {
                let offset = (address - cache.start) as usize;
                return Some(cache.write_ptr.wrapping_add(offset));
            }
        }
        None
    }

    fn invalidate_fast_cache(&mut self) {
        self.cached_region_index = None;
        self.cached_region_indices = [None; JIT_FAST_CACHE_ENTRIES];
        self.jit_fast_caches = [JitFastMemoryCache::default(); JIT_FAST_CACHE_ENTRIES];
    }

    fn remember_region_index(&mut self, index: usize) {
        self.cached_region_index = Some(index);

        let existing_position = self
            .cached_region_indices
            .iter()
            .position(|cached| *cached == Some(index))
            .unwrap_or(JIT_FAST_CACHE_ENTRIES - 1);
        for slot in (1..=existing_position).rev() {
            self.cached_region_indices[slot] = self.cached_region_indices[slot - 1];
        }
        self.cached_region_indices[0] = Some(index);
        self.rebuild_jit_fast_caches();
    }

    fn rebuild_jit_fast_caches(&mut self) {
        let indices = self.cached_region_indices;
        let mut caches = [JitFastMemoryCache::default(); JIT_FAST_CACHE_ENTRIES];
        for (slot, index) in indices.into_iter().flatten().enumerate() {
            if let Some(region) = self.regions.get_mut(index) {
                caches[slot] = JitFastMemoryCache {
                    start: region.start,
                    end: region.end(),
                    read_ptr: region.data.as_ptr(),
                    write_ptr: if region.is_execute() {
                        std::ptr::null_mut()
                    } else {
                        region.data.as_mut_ptr()
                    },
                };
            }
        }
        self.jit_fast_caches = caches;
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
        let offset = (address - region.start) as usize;
        let changes_code = region.is_execute();
        region.data[offset] = value;
        if changes_code {
            self.bump_code_version();
        }
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
        if len <= 8 {
            if let Some(region) = self.find_region_range(address, len as usize) {
                let offset = (address - region.start) as usize;
                return Ok(read_le_nbytes(&region.data, offset, len as usize));
            }
        }

        let mut result = Vec::new();
        for addr in address..address + len {
            result.push(self.read_byte(addr)?)
        }

        Ok(result
            .iter()
            .enumerate()
            .fold(0u64, |res, (idx, val)| res + (u64::from(*val) << (8 * idx))))
    }

    pub fn read_nbytes_cached(&mut self, address: u64, len: u64) -> Result<u64, MemoryError> {
        if len <= 8 {
            if let Some(index) = self.cached_region_index_for_range(address, len as usize) {
                let region = &self.regions[index];
                let offset = (address - region.start) as usize;
                return Ok(read_le_nbytes(&region.data, offset, len as usize));
            }
        }

        self.read_nbytes(address, len)
    }

    pub fn read_u8_cached(&mut self, address: u64) -> Result<u8, MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, 1) {
            let region = &self.regions[index];
            let offset = (address - region.start) as usize;
            return Ok(region.data[offset]);
        }

        self.read_byte(address)
    }

    pub fn read_u16_cached(&mut self, address: u64) -> Result<u16, MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, 2) {
            let region = &self.regions[index];
            let offset = (address - region.start) as usize;
            return Ok(read_le_nbytes(&region.data, offset, 2) as u16);
        }

        self.read_nbytes(address, 2).map(|value| value as u16)
    }

    pub fn read_u32_cached(&mut self, address: u64) -> Result<u32, MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, 4) {
            let region = &self.regions[index];
            let offset = (address - region.start) as usize;
            return Ok(read_le_nbytes(&region.data, offset, 4) as u32);
        }

        self.read_word(address)
    }

    pub fn read_u64_cached(&mut self, address: u64) -> Result<u64, MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, 8) {
            let region = &self.regions[index];
            let offset = (address - region.start) as usize;
            return Ok(read_le_nbytes(&region.data, offset, 8));
        }

        self.read_nbytes(address, 8)
    }

    pub fn write_nbytes(&mut self, address: u64, value: u64, len: u64) -> Result<(), MemoryError> {
        if len <= 8 {
            if let Some(region) = self.find_region_range_mut(address, len as usize) {
                let offset = (address - region.start) as usize;
                let changes_code = region.is_execute();
                write_le_nbytes(&mut region.data, offset, value, len as usize);
                if changes_code {
                    self.bump_code_version();
                }
                return Ok(());
            }
        }

        for (idx, addr) in (address..address + len).enumerate() {
            self.write_byte(addr, ((value >> (8 * idx)) & 0xFF) as u8)?;
        }

        Ok(())
    }

    pub fn write_nbytes_cached(
        &mut self,
        address: u64,
        value: u64,
        len: u64,
    ) -> Result<(), MemoryError> {
        if len <= 8 {
            if let Some(index) = self.cached_region_index_for_range(address, len as usize) {
                let changes_code = {
                    let region = &mut self.regions[index];
                    let offset = (address - region.start) as usize;
                    let changes_code = region.is_execute();
                    write_le_nbytes(&mut region.data, offset, value, len as usize);
                    changes_code
                };
                if changes_code {
                    self.bump_code_version();
                }
                return Ok(());
            }
        }

        self.write_nbytes(address, value, len)
    }

    pub fn write_u8_cached(&mut self, address: u64, value: u8) -> Result<(), MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, 1) {
            let changes_code = {
                let region = &mut self.regions[index];
                let offset = (address - region.start) as usize;
                let changes_code = region.is_execute();
                region.data[offset] = value;
                changes_code
            };
            if changes_code {
                self.bump_code_version();
            }
            return Ok(());
        }

        self.write_byte(address, value)
    }

    pub fn write_u16_cached(&mut self, address: u64, value: u16) -> Result<(), MemoryError> {
        self.write_cached_scalar(address, u64::from(value), 2)
    }

    pub fn write_u32_cached(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        self.write_cached_scalar(address, u64::from(value), 4)
    }

    pub fn write_u64_cached(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        self.write_cached_scalar(address, value, 8)
    }

    fn write_cached_scalar(
        &mut self,
        address: u64,
        value: u64,
        len: usize,
    ) -> Result<(), MemoryError> {
        if let Some(index) = self.cached_region_index_for_range(address, len) {
            let changes_code = {
                let region = &mut self.regions[index];
                let offset = (address - region.start) as usize;
                let changes_code = region.is_execute();
                write_le_nbytes(&mut region.data, offset, value, len);
                changes_code
            };
            if changes_code {
                self.bump_code_version();
            }
            return Ok(());
        }

        self.write_nbytes(address, value, len as u64)
    }

    pub(crate) fn direct_write_ptr_range(
        &mut self,
        address: u64,
        len: u64,
    ) -> Result<*mut u8, MemoryError> {
        let len = usize::try_from(len).map_err(|_| MemoryError::InvalidAddress(address))?;
        if let Some(ptr) = self.fast_cached_write_ptr_range(address, len) {
            return Ok(ptr);
        }
        if let Some(index) = self.cached_region_index_for_range(address, len) {
            let changes_code = {
                let region = &self.regions[index];
                region.is_execute()
            };
            let region = &mut self.regions[index];
            let offset = (address - region.start) as usize;
            let ptr = region.data.as_mut_ptr().wrapping_add(offset);
            if changes_code {
                self.bump_code_version();
            }
            return Ok(ptr);
        }

        let (ptr, changes_code) = {
            let region = self
                .find_region_range_mut(address, len)
                .ok_or(MemoryError::InvalidAddress(address))?;
            let offset = (address - region.start) as usize;
            let ptr = region.data.as_mut_ptr().wrapping_add(offset);
            (ptr, region.is_execute())
        };
        if changes_code {
            self.bump_code_version();
        }
        Ok(ptr)
    }

    pub(crate) fn direct_read_ptr_range(
        &mut self,
        address: u64,
        len: u64,
    ) -> Result<*const u8, MemoryError> {
        let len = usize::try_from(len).map_err(|_| MemoryError::InvalidAddress(address))?;
        if let Some(ptr) = self.fast_cached_read_ptr_range(address, len) {
            return Ok(ptr);
        }
        if let Some(index) = self.cached_region_index_for_range(address, len) {
            let region = &self.regions[index];
            let offset = (address - region.start) as usize;
            return Ok(region.data.as_ptr().wrapping_add(offset));
        }
        Err(MemoryError::InvalidAddress(address))
    }

    pub(crate) fn read_slice_range(&self, address: u64, len: u64) -> Result<&[u8], MemoryError> {
        let len = usize::try_from(len).map_err(|_| MemoryError::InvalidAddress(address))?;
        let region = self
            .find_region_range(address, len)
            .ok_or(MemoryError::InvalidAddress(address))?;
        let offset = (address - region.start) as usize;
        Ok(&region.data[offset..offset + len])
    }

    pub(crate) fn write_slice_range(
        &mut self,
        address: u64,
        len: u64,
    ) -> Result<&mut [u8], MemoryError> {
        let len = usize::try_from(len).map_err(|_| MemoryError::InvalidAddress(address))?;
        let index = self
            .cached_region_index_for_range(address, len)
            .ok_or(MemoryError::InvalidAddress(address))?;
        let changes_code = self.regions[index].is_execute();
        if changes_code {
            self.bump_code_version();
        }
        let region = &mut self.regions[index];
        let offset = (address - region.start) as usize;
        Ok(&mut region.data[offset..offset + len])
    }

    pub fn read_word(&self, address: u64) -> Result<u32, MemoryError> {
        if let Some(region) = self.find_region_range(address, 4) {
            let offset = (address - region.start) as usize;
            return Ok(read_le_nbytes(&region.data, offset, 4) as u32);
        }

        let b0 = self.read_byte(address)? as u32;
        let b1 = self.read_byte(address + 1)? as u32;
        let b2 = self.read_byte(address + 2)? as u32;
        let b3 = self.read_byte(address + 3)? as u32;
        Ok(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    pub fn write_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        if let Some(region) = self.find_region_range_mut(address, 4) {
            let offset = (address - region.start) as usize;
            let changes_code = region.is_execute();
            write_le_nbytes(&mut region.data, offset, value.into(), 4);
            if changes_code {
                self.bump_code_version();
            }
            return Ok(());
        }

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

    fn bump_code_version(&mut self) {
        self.code_version = self.code_version.wrapping_add(1);
    }
}

fn read_le_nbytes(data: &[u8], offset: usize, len: usize) -> u64 {
    // Callers range-check before reaching this helper; unaligned scalar loads
    // avoid calling into memcpy/memmove for the JIT's hot stack accesses.
    unsafe {
        let ptr = data.as_ptr().add(offset);
        match len {
            1 => u64::from(ptr.read()),
            2 => u64::from(u16::from_le(ptr.cast::<u16>().read_unaligned())),
            4 => u64::from(u32::from_le(ptr.cast::<u32>().read_unaligned())),
            8 => u64::from_le(ptr.cast::<u64>().read_unaligned()),
            _ => {
                let mut value = 0;
                for idx in 0..len {
                    value |= u64::from(ptr.add(idx).read()) << (8 * idx);
                }
                value
            }
        }
    }
}

fn write_le_nbytes(data: &mut [u8], offset: usize, value: u64, len: usize) {
    // Callers range-check before reaching this helper; unaligned scalar stores
    // avoid calling into memcpy/memmove for the JIT's hot stack accesses.
    unsafe {
        let ptr = data.as_mut_ptr().add(offset);
        match len {
            1 => ptr.write(value as u8),
            2 => ptr.cast::<u16>().write_unaligned((value as u16).to_le()),
            4 => ptr.cast::<u32>().write_unaligned((value as u32).to_le()),
            8 => ptr.cast::<u64>().write_unaligned(value.to_le()),
            _ => {
                for idx in 0..len {
                    ptr.add(idx).write(((value >> (8 * idx)) & 0xff) as u8);
                }
            }
        }
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

#[derive(Debug, Clone)]
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

    pub fn is_execute(&self) -> bool {
        self.flags & 1 == 1
    }

    fn end(&self) -> u64 {
        self.start + self.size
    }

    pub fn extend(&mut self, addition: u64) {
        self.data.extend(vec![0u8; addition as usize]);
        self.size += addition;
    }
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Invalid address: 0x{0:016x}")]
    InvalidAddress(u64),
    #[error("Permission denied at address: 0x{0:X}")]
    PermissionDenied(u64),
    #[error("Region overlap at address: 0x{0:X}")]
    RegionOverlap(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_break_starts_after_loaded_segments() {
        let mut ram = Ram::new();
        ram.add_region(MemoryRegion::new_with_flags(
            0x10000,
            0x1000,
            vec![0; 0x1000],
            1,
        ))
        .unwrap();
        ram.add_region(MemoryRegion::new(0x12000, 0x1000, vec![0; 0x1000]))
            .unwrap();

        ram.set_program_break_base(0x12345);

        assert_eq!(ram.program_break(), 0x13000);
        assert_eq!(ram.set_program_break(0x14020).unwrap(), 0x14020);
        assert!(ram.write_byte(0x12010, 0xbb).is_ok());
        assert!(ram.write_byte(0x13fff, 0xaa).is_ok());
    }

    #[test]
    fn mmap_does_not_use_stack_as_the_next_base() {
        let mut ram = Ram::new();
        ram.add_region(MemoryRegion::new(
            0x7fff_ffff_fff0 - PAGE_SIZE,
            PAGE_SIZE,
            vec![0; PAGE_SIZE as usize],
        ))
        .unwrap();

        let first = ram.mmap_anonymous(None, 1).unwrap();
        let second = ram.mmap_anonymous(None, PAGE_SIZE).unwrap();

        assert_eq!(first, MMAP_BASE);
        assert_eq!(second, MMAP_BASE + PAGE_SIZE);
    }

    #[test]
    fn munmap_can_split_an_existing_mapping() {
        let mut ram = Ram::new();
        let addr = ram.mmap_anonymous(None, PAGE_SIZE * 3).unwrap();
        ram.write_byte(addr, 1).unwrap();
        ram.write_byte(addr + PAGE_SIZE, 2).unwrap();
        ram.write_byte(addr + PAGE_SIZE * 2, 3).unwrap();

        ram.munmap(addr + PAGE_SIZE, PAGE_SIZE).unwrap();

        assert_eq!(ram.read_byte(addr).unwrap(), 1);
        assert!(ram.read_byte(addr + PAGE_SIZE).is_err());
        assert_eq!(ram.read_byte(addr + PAGE_SIZE * 2).unwrap(), 3);
    }

    #[test]
    fn code_version_tracks_executable_memory_changes_only() {
        let mut ram = Ram::new();
        ram.add_region(MemoryRegion::new_with_flags(
            0x1000,
            PAGE_SIZE,
            vec![0; PAGE_SIZE as usize],
            1,
        ))
        .unwrap();
        let loaded_version = ram.code_version();
        ram.add_region(MemoryRegion::new(
            0x3000,
            PAGE_SIZE,
            vec![0; PAGE_SIZE as usize],
        ))
        .unwrap();
        assert_eq!(ram.code_version(), loaded_version);

        ram.write_byte(0x3000, 1).unwrap();
        assert_eq!(ram.code_version(), loaded_version);

        ram.write_byte(0x1000, 1).unwrap();
        assert_eq!(ram.code_version(), loaded_version + 1);
    }

    #[test]
    fn direct_pointer_ranges_reuse_jit_fast_cache() {
        let mut ram = Ram::new();
        ram.add_region(MemoryRegion::new(0x1000, 32, vec![0; 32]))
            .unwrap();

        let read_ptr = ram.direct_read_ptr_range(0x1004, 8).unwrap();
        let cache = ram.jit_fast_caches[0];
        assert_eq!(read_ptr, unsafe { cache.read_ptr.add(4) });

        let write_ptr = ram.direct_write_ptr_range(0x1008, 8).unwrap();
        assert_eq!(write_ptr, unsafe { cache.write_ptr.add(8) });
        unsafe {
            write_ptr.write(0xab);
        }

        assert_eq!(ram.read_byte(0x1008).unwrap(), 0xab);
    }
}
