# minimal_tls_check.S
# RISC-V Assembly Program to check for PT_TLS segment

    .section .text
    .global _start
_start:
    # Save the initial stack pointer (optional)
    mv x31, sp

    # Retrieve argc from the stack
    ld a0, 0(sp)          # a0 = argc

    # Calculate argv address
    addi a1, sp, 8        # a1 = sp + 8 (argv[0])

    # Calculate envp address
    slli t0, a0, 3        # t0 = argc * 8 (size of argv entries)
    add a2, a1, t0        # a2 = a1 + t0 (argv + argc*8)
    addi a2, a2, 8        # a2 = a2 + 8 (skip NULL terminator)

    # Find auxv (envp ends with NULL)
find_auxv:
    ld t1, 0(a2)
    beqz t1, auxv_found
    addi a2, a2, 8        # Move to next envp entry
    j find_auxv
auxv_found:
    addi a2, a2, 8        # Skip NULL after envp
    # Now a2 points to auxv

    # Initialize registers to store AT_PHDR, AT_PHENT, AT_PHNUM
    li t3, 0              # phdr_addr
    li t4, 0              # phent_size
    li t5, 0              # phnum

    # Search auxv for AT_PHDR, AT_PHENT, AT_PHNUM
search_auxv:
    ld t1, 0(a2)          # t1 = auxv_type
    ld t2, 8(a2)          # t2 = auxv_value
    beqz t1, auxv_end     # If auxv_type == AT_NULL (0), end
    li t0, 3              # AT_PHDR
    beq t1, t0, got_phdr
    li t0, 4              # AT_PHENT
    beq t1, t0, got_phent
    li t0, 5              # AT_PHNUM
    beq t1, t0, got_phnum
    addi a2, a2, 16       # Move to next auxv entry
    j search_auxv

got_phdr:
    mv t3, t2             # phdr_addr = auxv_value
    addi a2, a2, 16
    j search_auxv

got_phent:
    mv t4, t2             # phent_size = auxv_value
    addi a2, a2, 16
    j search_auxv

got_phnum:
    mv t5, t2             # phnum = auxv_value
    addi a2, a2, 16
    j search_auxv

auxv_end:
    # Check if we have all required values
    beqz t3, auxv_error   # If phdr_addr == 0, error
    beqz t4, auxv_error   # If phent_size == 0, error
    beqz t5, auxv_error   # If phnum == 0, error

    # Iterate over program headers to find PT_TLS
    li t6, 0              # i = 0
    mv s0, t3             # s0 = phdr_addr (phdr_ptr)

ph_loop:
    bgeu t6, t5, not_found  # if i >= phnum, PT_TLS not found
    ld t0, 0(s0)          # t0 = p_type
    li t1, 7              # PT_TLS
    beq t0, t1, found_tls
    add s0, s0, t4        # s0 = s0 + phent_size (phdr_ptr += phent_size)
    addi t6, t6, 1        # i++
    j ph_loop

found_tls:
    # Output "Found PT_TLS\n"
    la a1, msg_found      # a1 = address of message
    li a2, 12             # a2 = message length
    li a0, 1              # a0 = stdout (fd = 1)
    jal write_stdout
    j exit

not_found:
    # Output "PT_TLS not found\n"
    la a1, msg_not_found  # a1 = address of message
    li a2, 17             # a2 = message length
    li a0, 1              # a0 = stdout (fd = 1)
    jal write_stdout
    j exit

auxv_error:
    # Output "Error reading auxv\n"
    la a1, msg_auxv_error # a1 = address of message
    li a2, 19             # a2 = message length
    li a0, 1              # a0 = stdout (fd = 1)
    jal write_stdout
    j exit

write_stdout:
    # a0 = fd
    # a1 = message address
    # a2 = message length
    li a7, 64             # syscall number for write in RISC-V Linux
    ecall
    ret

exit:
    # Exit the program
    li a7, 93             # syscall number for exit in RISC-V Linux
    li a0, 0              # exit code 0
    ecall

    .section .tdata,"awT",@progbits
    .align 8
    .global tls_variable
tls_variable:
    .quad 42              # Initialize thread-local variable

    .section .rodata
msg_found:
    .ascii "Found PT_TLS\n"
msg_not_found:
    .ascii "PT_TLS not found\n"
msg_auxv_error:
    .ascii "Error reading auxv\n"

