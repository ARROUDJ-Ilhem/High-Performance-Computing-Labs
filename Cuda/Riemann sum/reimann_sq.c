#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define N 1000000000      // Nombre de sous-intervales (augmenter pour plus de précision)

// Fonction f(x)
double f(double x) {
    return exp(-x * x);
}

// Somme de Reimann
double somme_reimann(double a, double b, long int n) {
    long double delta_x = (b - a) / n; //Le pas
    long double somme = 0.0;

    for (long int i = 0; i < n; i++) {
        long double x_i = a + i * delta_x;
        somme += f(x_i) * delta_x;
    }

    return somme;
}

int main() {
    srand(time(NULL));
    clock_t t;
    double a = 0.0;  // Borne inf de l'integral
    double b = 1.0;  // Borne sup de l'integral
    
    t = clock();
    
    long double resultat = somme_reimann(a, b, N);
    
    printf("Temps d'execution: %f\n",((double)(clock()-t)/CLOCKS_PER_SEC));
    printf("Somme de Reimann (integral approximé): %Lf\n", resultat);

    return 0;
}
