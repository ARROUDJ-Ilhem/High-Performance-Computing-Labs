#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 16, chunksize = 4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Erreur : >= 2 processus requis.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int num_chunks = n / chunksize;
    int *chunk = malloc(chunksize * sizeof(int));

    // ── MAÎTRE ──────────────────────────────────────────────────────────────
    if (rank == 0) {
        int *data    = malloc(n * sizeof(int));
        int *results = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) data[i] = i + 1;

        int next_chunk   = 0;   // prochain chunk à distribuer
        int active_slaves = size - 1;
        int first_requests = size - 1; // premières demandes à ignorer (pas de résultat)

        while (active_slaves > 0) {
            MPI_Status status;
            int offset;

            // Réception : première demande (tag=0, pas de résultat)
            //             ou résultat + nouvelle demande (tag=offset du chunk traité)
            MPI_Recv(chunk, chunksize, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);

            int source = status.MPI_SOURCE;

            // Stocker le résultat si ce n'est pas la première demande
            if (first_requests > 0) {
                first_requests--;
            } else {
                offset = status.MPI_TAG; // tag = index du chunk dans data[]
                for (int i = 0; i < chunksize; i++)
                    results[offset * chunksize + i] = chunk[i];
            }

            // Envoyer un nouveau chunk ou l'ordre de fin
            if (next_chunk < num_chunks) {
                // tag = index du chunk, pour que l'esclave le renvoie ensuite
                MPI_Send(&data[next_chunk * chunksize], chunksize, MPI_INT,
                         source, next_chunk, MPI_COMM_WORLD);
                next_chunk++;
            } else {
                // Ordre de fin : tag = -1
                MPI_Send(chunk, 0, MPI_INT, source, -1, MPI_COMM_WORLD);
                active_slaves--;
            }
        }

        printf("Résultats (carré de 1..%d) :\n", n);
        for (int i = 0; i < n; i++) printf("%d ", results[i]);
        printf("\n");

        free(data); free(results);
    }

    // ── ESCLAVES ─────────────────────────────────────────────────────────────
    else {
        // Première demande : tag=0, contenu ignoré par le maître
        MPI_Send(chunk, chunksize, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            MPI_Status status;
            MPI_Recv(chunk, chunksize, MPI_INT, 0, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == -1) break; // ordre de fin

            int chunk_index = status.MPI_TAG;

            // Traitement : mise au carré
            for (int i = 0; i < chunksize; i++)
                chunk[i] = chunk[i] * chunk[i];

            // Renvoi du résultat + nouvelle demande implicite
            // tag = index du chunk pour que le maître sache où le stocker
            MPI_Send(chunk, chunksize, MPI_INT, 0, chunk_index, MPI_COMM_WORLD);
        }
    }

    free(chunk);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
