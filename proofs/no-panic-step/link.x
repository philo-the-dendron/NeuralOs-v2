/* The memory map of a program that is linked and never run: the link is
   the verdict (build.sh). After proofs/qemu-riscv-leg-a/link.x, for rv32. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  .text : {
    KEEP(*(.text._start))
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(4) {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
    *(.eh_frame .eh_frame.*)
  } > RAM

  .data : ALIGN(4) {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    *(.data .data.*)
  } > RAM

  .bss (NOLOAD) : ALIGN(4) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
  } > RAM

  .got : ALIGN(4) {
    *(.got .got.*)
  } > RAM
}
