#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    Point *points;
    int start;
    int end;
    Point target;
    Point nearest;
    double min_dist;
} ThreadData;

double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void *nearest_neighbor_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    data->nearest = data->points[data->start];
    data->min_dist = distance(data->nearest, data->target);

    for (int i = data->start + 1; i < data->end; i++) {
        double dist = distance(data->points[i], data->target);
        if (dist < data->min_dist) {
            data->min_dist = dist;
            data->nearest = data->points[i];
        }
    }
    return NULL;
}

Point nearest_neighbor_parallel(Point *points, int n, Point target, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int segment_size = n / num_threads;

    // Initialiser et lancer les threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].points = points;
        thread_data[i].start = i * segment_size;
        thread_data[i].end = (i == num_threads - 1) ? n : (i + 1) * segment_size;
        thread_data[i].target = target;
        pthread_create(&threads[i], NULL, nearest_neighbor_thread, &thread_data[i]);
    }

    // Attendre la fin de chaque thread et trouver le plus proche voisin global
    Point nearest = points[0];
    double min_dist = distance(points[0], target);
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        if (thread_data[i].min_dist < min_dist) {
            min_dist = thread_data[i].min_dist;
            nearest = thread_data[i].nearest;
        }
    }

    return nearest;
}

int main() {
    int n = 10000000; // 10 millions de points
    int num_threads;
    printf("Entrez le nombre de threads : ");
    scanf("%d", &num_threads);

    Point *points = malloc(n * sizeof(Point));

    // Initialisation des points avec des valeurs aléatoires
    srand(42);
    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000;
        points[i].y = rand() % 1000;
    }

    Point target = {500, 500}; // Point cible

    clock_t start = clock();
    Point nearest = nearest_neighbor_parallel(points, n, target, num_threads);
    clock_t end = clock();

    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Plus proche voisin (parallèle): (%.2f, %.2f)\n", nearest.x, nearest.y);
    printf("Durée d'exécution (parallèle) avec %d threads : %.6f secondes\n", num_threads, execution_time);

    free(points);
    return 0;
}
