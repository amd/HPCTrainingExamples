! LCOMPILER-2446 reproducer (reported by Toshihiro Kato, NEC).
! amdflang mis-parses a named DO construct whose construct name is `block`
! (block: do ... end do block) when it is the first construct of an OpenMP
! region's structured block, reporting "expected OpenMP construct". Renaming
! the construct (e.g. jloop:) or dropping the OpenMP directive compiles fine.
! The failure is at compile time, so this test checks that it parses.
program named_block_do
  implicit none
  integer :: i, s
  s = 0
  ! The named DO `block:` is the first construct of the target region's
  ! structured block -- this is what amdflang mis-parses.
  !$omp target teams map(tofrom:s)
  block: do i = 1, 100
     s = s + 1
  end do block
  !$omp end target teams
  print *, 's =', s
end program named_block_do
