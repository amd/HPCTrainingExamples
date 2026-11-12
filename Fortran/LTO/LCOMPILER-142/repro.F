C     LCOMPILER-142 reproducer (from Mashy Green, UCL).
C     Under -flto, amdflang truncates the ELF st_size and .lbss section size
C     of a COMMON block whose declared size exceeds 2**32 bytes to size mod
C     2**32, while element-address codegen still uses the full 64-bit offset.
C     The first store past the truncated allocation then lands in unmapped
C     memory and segfaults with no diagnostic.
C
C     Common block /big/ emits the symbol big_. Here it is
C     4 * 1,500,000,000 = 6,000,000,000 bytes = 0x1_65A0BC00; the low 32 bits
C     are 0x65A0BC00. The accompanying test asserts readelf reports the full
C     64-bit size, not the truncated one.
      program repro
      implicit none
      integer, parameter :: n = 1500000000
      real big(n)
      common /big/ big
      big(n) = 1.0
      print *, 'big(n)=', big(n)
      end
