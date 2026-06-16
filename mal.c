// synthetic_crypto.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/random.h>

#define BUFFER_SIZE 4096
#define TEMP_FILE "/tmp/cryptotrace_test.bin"

int main() {
    unsigned char buffer[BUFFER_SIZE];
    unsigned char ciphertext[BUFFER_SIZE + 32];
    unsigned char key[32];
    unsigned char iv[16];

    printf("[+] Starting synthetic crypto workload...\n");

    // Generate random key and IV
    if (getrandom(key, sizeof(key), 0) != sizeof(key) ||
        getrandom(iv, sizeof(iv), 0) != sizeof(iv)) {
        perror("getrandom");
        return 1;
    }

    // Create temporary file
    int fd = open(TEMP_FILE, O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Generate random data and write to file
    for (int i = 0; i < 100; i++) {
        if (getrandom(buffer, BUFFER_SIZE, 0) != BUFFER_SIZE) {
            perror("getrandom");
            close(fd);
            return 1;
        }

        write(fd, buffer, BUFFER_SIZE);
    }

    fsync(fd);
    lseek(fd, 0, SEEK_SET);

    // Initialize AES context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        fprintf(stderr, "Failed to create cipher context\n");
        close(fd);
        return 1;
    }

    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);

    // Read file and encrypt contents (in memory only)
    ssize_t bytes_read;
    int outlen;

    while ((bytes_read = read(fd, buffer, BUFFER_SIZE)) > 0) {
        EVP_EncryptUpdate(
            ctx,
            ciphertext,
            &outlen,
            buffer,
            bytes_read
        );
    }

    EVP_EncryptFinal_ex(ctx, ciphertext, &outlen);

    EVP_CIPHER_CTX_free(ctx);

    close(fd);

    // Remove temp file
    unlink(TEMP_FILE);

    printf("[+] Synthetic crypto workload completed safely.\n");

    return 0;
}