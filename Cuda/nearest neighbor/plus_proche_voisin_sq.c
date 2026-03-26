
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    double x;
    double y;
} Point;

double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

Point nearest_neighbor_sequential(Point *points, int n, Point target) {
    Point nearest = points[0];
    double min_dist = distance(points[0], target);

    for (int i = 1; i < n; i++) {
        double dist = distance(points[i], target);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = points[i];
        }
    }
    return nearest;
}

int main() {
    int n = 10000000; // 10 millions de points
    Point *points = malloc(n * sizeof(Point));

    // Initialisation des points avec des valeurs aléatoires
    srand(42);
    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000;
        points[i].y = rand() % 1000;
    }

    Point target = {500, 500}; // Point cible

    clock_t start = clock();
    Point nearest = nearest_neighbor_sequential(points, n, target);
    clock_t end = clock();

    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Plus proche voisin (séquentiel): (%.2f, %.2f)\n", nearest.x, nearest.y);
    printf("Durée d'exécution (séquentielle) : %.6f secondes\n", execution_time);

    free(points);
    return 0;
}
