program media_notas
    implicit none
    real :: notas(6)
    real :: media
    integer :: i

    ! Definir as notas
    notas = (/ 8.0, 9.0, 7.5, 10.0, 6.0, 9.5/)

    ! Calcular a média
    media = sum(notas) / size(notas)

    ! Exibir o resultado
    print *, "A média das notas é:", media

end program media_notas
