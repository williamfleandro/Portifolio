program knn_regressor

  implicit none

  integer, parameter :: k = 2
  integer, parameter :: num_train = 5
  integer, parameter :: num_test = 1
  integer, parameter :: num_features = 1

  real, dimension(num_train, num_features) :: train_data
  real, dimension(num_train) :: train_target
  real, dimension(num_test, num_features) :: test_data
  real, dimension(num_test) :: predicted_target

  integer :: i, j, l
  real :: distance
  real, dimension(num_train) :: distances
  integer, dimension(num_train) :: indices
  real :: sum_target
  real :: start_time, end_time

  ! Inicializar dados de treino
  train_data(1, 1) = 1.0
  train_data(2, 1) = 2.0
  train_data(3, 1) = 3.0
  train_data(4, 1) = 4.0
  train_data(5, 1) = 5.0

  train_target = (/ 2.0, 4.0, 6.0, 8.0, 10.0 /)

  ! Inicializar dados de teste
  test_data(1, 1) = 3.5

  ! Início da medição de tempo
  call cpu_time(start_time)

  ! Loop para cada ponto de teste (apenas 1 neste caso)
  do i = 1, num_test

    ! Calcular distâncias entre o ponto de teste e todos os pontos de treinamento
    do j = 1, num_train
      distance = 0.0
      do l = 1, num_features
        distance = distance + (test_data(i, l) - train_data(j, l))**2
      end do
      distances(j) = sqrt(distance)
      indices(j) = j
    end do

    ! Encontrar os k vizinhos mais próximos
    call sort_with_indices(distances, indices, num_train)

    ! Calcular a média dos valores alvo dos k vizinhos
    sum_target = 0.0
    do j = 1, k
      sum_target = sum_target + train_target(indices(j))
    end do
    predicted_target(i) = sum_target / k

  end do

  ! Fim da medição de tempo
  call cpu_time(end_time)

  ! Imprimir resultados e tempo de execução
  print *, "Previsão: ", predicted_target(1)
  print *, "Tempo de execução: ", end_time - start_time, " segundos"

contains

  ! Subrotina para ordenar um array e acompanhar os índices
  subroutine sort_with_indices(array, indices, n)
    real, dimension(n), intent(inout) :: array
    integer, dimension(n), intent(inout) :: indices
    integer, intent(in) :: n
    integer :: i, j, temp_index
    real :: temp

    do i = 1, n - 1
      do j = i + 1, n
        if (array(i) > array(j)) then
          temp = array(i)
          array(i) = array(j)
          array(j) = temp
          temp_index = indices(i)
          indices(i) = indices(j)
          indices(j) = temp_index
        end if
      end do
    end do
  end subroutine sort_with_indices

end program knn_regressor