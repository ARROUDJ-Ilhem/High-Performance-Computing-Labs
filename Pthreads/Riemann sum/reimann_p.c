#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4 // Nombre de threads
#define N 1000000000      // Nombre de sous-intervales (augmenter pour plus de précision)

typedef struct {
    double a, b;   // Interval [a, b]
    long int n;         // Nombre de sous-intervales
    long int debut, fin; // Sous-intervale
} Chunck;

// Fonction f(x)
double f(double x) {
    return exp(-x * x);
}

// Somme de Reimann partielle
void* somme_reimann_partielle(void* arg) {
    Chunck* donnee = (Chunck*)arg;
    long double delta_x = (donnee->b - donnee->a) / donnee->n; //Le pas

    long double* somme = malloc(sizeof(double));
    if (somme == NULL) {
        perror("Allocation de mémoire échouée");
        pthread_exit(NULL);
    }

    *somme=0.0;
    for (long int i = donnee->debut; i < donnee->fin; i++) {
        long double x_i = donnee->a + i * delta_x;
        *somme += f(x_i) * delta_x;
    }

    pthread_exit(somme);
}

int main() {
    srand(time(NULL));
    clock_t t;
    double a = 0.0;      // Borne inf de l'integral
    double b = 1.0;      // Borne sup de l'integral
    pthread_t threads[NUM_THREADS];
    Chunck chunck[NUM_THREADS];

    long int intervals_par_thread = N / NUM_THREADS;

    t = clock();

    for (int t = 0; t < NUM_THREADS; t++) {
        chunck[t].a = a;
        chunck[t].b = b;
        chunck[t].n = N;
        chunck[t].debut = t * intervals_par_thread;

        chunck[t].fin = (t + 1) * intervals_par_thread;
        if(t == NUM_THREADS - 1) chunck[t].fin = N; //En cas ou n n'est pas divisble par NUM_THREADS, le dérnier thread prend le travail restant

        if (pthread_create(&threads[t], NULL, somme_reimann_partielle, &chunck[t]) != 0) {
            perror("Error lors de la création du thread");
            exit(1);
        }
    }
    
    long double resultat = 0.0;
    long double* somme_partielle;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], (void**)&somme_partielle);
        resultat += *somme_partielle;

        free(somme_partielle);
    }
    
    printf("Temps d'execution: %f\n",((double)(clock()-t)/CLOCKS_PER_SEC));
    printf("Somme de Reimann (integral approximé): %Lf\n", resultat);

    return 0;
}

