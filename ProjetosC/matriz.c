#include <stdio.h>
#include <stdlib.h>

int *criarMatriz(int n, int m) {
    int *matriz = malloc(n * m * sizeof(int));
    if (matriz == NULL) {
        printf("Erro ao alocar memória!\n");
        return NULL;
    }

    int valor = 0;
    for (int i = 0; i < n * m; i++) {
        valor = valor + i;
        matriz[i] = valor;
    }

    return matriz;
}

void exibirMatriz(int* matriz, int n, int m) {
    puts("Funcao em C");
    for (int i = 0; i < n; i++){
        if(i%m != 0 || i == 0 )
        printf("%d", matriz[i]);
        else{
            printf("\n");
            puts(" ");
        }
    }
}


    // int main() {
    //     int linhas = 3, colunas = 3;
    //     int *matriz = criarMatriz(linhas, colunas);

    //     if (matriz != NULL) {
    //         printf("Matriz criada:\n");
    //         for (int i = 0; i < linhas; i++) {
    //             for (int j = 0; j < colunas; j++) {
    //                 printf("%d ", matriz[i * colunas + j]);
    //             }
    //             printf("\n");
    //         }
    //         free(matriz);
    //     }

    //     return 0;
    // }
