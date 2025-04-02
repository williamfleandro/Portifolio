import m
from time import perf_counter

start = perf_counter()

mm = m.PyMatrix(20, 50, 1)
mm.exibir()

end = perf_counter()

del mm
print(f"Tempo de execução: {end - start:.6f} segundos.")

