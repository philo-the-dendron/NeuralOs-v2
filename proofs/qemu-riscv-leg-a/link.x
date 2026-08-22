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

  .rodata : ALIGN(8) {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
    *(.eh_frame .eh_frame.*)
  } > RAM

  .data : ALIGN(8) {
    _data_start = .;
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    *(.data .data.*)
    . = ALIGN(8);
    _data_end = .;
  } > RAM

  .bss (NOLOAD) : ALIGN(8) {
    _bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > RAM

  .got : ALIGN(8) {
    *(.got .got.*)
  } > RAM

  . = ALIGN(16);
  _stack_bottom = .;
  . += 64K;
  . = ALIGN(16);
  _stack_top = .;
  _end = .;
}
