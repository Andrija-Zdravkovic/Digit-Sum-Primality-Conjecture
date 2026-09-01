#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#ifndef TOTAL_LIMIT
#define TOTAL_LIMIT   1000000000000ULL   // 10^12
#endif
#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE  100000000ULL       // 100 million per segment
#endif
#define MAX_STEPS     10000              // same bailout as Python (counterexample flag)

static const char *CHECKPOINT_FILE  = "zdravkovic_checkpoint.json";
static const char *MILESTONES_JSON  = "zdravkovic_milestones.json";
static const char *MILESTONES_TXT   = "zdravkovic_milestones.txt";

// ---------------------------------------------------------
// 1. BASE PRIMES GENERATOR (up to sqrt(LIMIT) ~ 1,000,000)
// ---------------------------------------------------------
static uint64_t *get_base_primes(uint64_t limit, size_t *out_count) {
    uint64_t sqrt_limit = (uint64_t)sqrt((double)limit) + 1;
    uint8_t *is_prime = malloc((sqrt_limit + 1) * sizeof(uint8_t));
    memset(is_prime, 1, sqrt_limit + 1);
    is_prime[0] = 0;
    if (sqrt_limit >= 1) is_prime[1] = 0;

    uint64_t inner_sqrt = (uint64_t)sqrt((double)sqrt_limit) + 1;
    for (uint64_t i = 2; i <= inner_sqrt; i++) {
        if (is_prime[i]) {
            for (uint64_t j = i * i; j <= sqrt_limit; j += i) {
                is_prime[j] = 0;
            }
        }
    }

    size_t count = 0;
    for (uint64_t i = 2; i <= sqrt_limit; i++) if (is_prime[i]) count++;

    uint64_t *primes = malloc(count * sizeof(uint64_t));
    size_t idx = 0;
    for (uint64_t i = 2; i <= sqrt_limit; i++) if (is_prime[i]) primes[idx++] = i;

    free(is_prime);
    *out_count = count;
    return primes;
}

// ---------------------------------------------------------
// 2. SLIDING SEGMENT BIT SIEVE
// ---------------------------------------------------------
// sieve_bytes must have at least ceil(seg_len/8) bytes.
static void sieve_segment(uint8_t *sieve_bytes, uint64_t seg_start, uint64_t seg_len,
                           const uint64_t *base_primes, size_t n_base_primes) {
    size_t nbytes = (seg_len + 7) / 8;
    memset(sieve_bytes, 0xFF, nbytes);

    if (seg_start == 0) {
        sieve_bytes[0] &= 0xFC; // clear bit 0 (value 0) and bit 1 (value 1)
    }

    for (size_t k = 0; k < n_base_primes; k++) {
        uint64_t p = base_primes[k];
        uint64_t p_sq = p * p;
        if (p_sq >= seg_start + seg_len) break;

        uint64_t start_val = ((seg_start + p - 1) / p) * p;
        if (start_val < p_sq) start_val = p_sq;

        for (uint64_t composite = start_val; composite < seg_start + seg_len; composite += p) {
            uint64_t idx = composite - seg_start;
            sieve_bytes[idx >> 3] &= ~(1 << (idx & 7));
        }
    }
}

// ---------------------------------------------------------
// 3. SEQUENCE LOGIC & LOOKUPS
// ---------------------------------------------------------
static inline uint64_t get_digit_sum(uint64_t n) {
    uint64_t s = 0;
    while (n > 0) {
        s += n % 10;
        n /= 10;
    }
    return s;
}

static inline bool fallback_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

static inline bool is_prime_segmented(uint64_t n, uint64_t seg_start, uint64_t seg_len,
                                       const uint8_t *sieve_bytes) {
    if (n < seg_start || n >= seg_start + seg_len) {
        return fallback_is_prime(n);
    }
    uint64_t idx = n - seg_start;
    return (sieve_bytes[idx >> 3] & (1 << (idx & 7))) != 0;
}

// returns step count, or -1 if it blew past MAX_STEPS (counterexample candidate)
static int32_t test_number(uint64_t start_num, uint64_t seg_start, uint64_t seg_len,
                            const uint8_t *sieve_bytes) {
    if (start_num % 3 == 0) return 0;

    uint64_t current = start_num;
    int32_t steps = 0;

    while (true) {
        if (is_prime_segmented(current, seg_start, seg_len, sieve_bytes)) {
            return steps;
        }
        current += get_digit_sum(current);
        steps += 1;
        if (steps > MAX_STEPS) {
            return -1;
        }
    }
}

// ---------------------------------------------------------
// 4. PARALLEL WORKER FOR SEGMENT
// ---------------------------------------------------------
static void process_segment_parallel(uint64_t seg_start, uint64_t seg_len,
                                      const uint8_t *sieve_bytes, int16_t *steps_out) {
    #pragma omp parallel for schedule(dynamic, 4096)
    for (uint64_t i = 0; i < seg_len; i++) {
        uint64_t val = seg_start + i;
        if (val % 3 != 0 && val > 0) {
            int32_t r = test_number(val, seg_start, seg_len, sieve_bytes);
            steps_out[i] = (int16_t)r;
        } else {
            steps_out[i] = 0;
        }
    }
}

// ---------------------------------------------------------
// 5. MILESTONE LOGGING & CHECKPOINTS
// ---------------------------------------------------------
typedef struct {
    uint64_t number;
    int steps;
    char timestamp[32];
} Milestone;

static Milestone *milestones = NULL;
static size_t milestones_count = 0;
static size_t milestones_cap = 0;

static void milestones_push(uint64_t number, int steps, const char *timestamp) {
    if (milestones_count == milestones_cap) {
        milestones_cap = milestones_cap ? milestones_cap * 2 : 16;
        milestones = realloc(milestones, milestones_cap * sizeof(Milestone));
    }
    milestones[milestones_count].number = number;
    milestones[milestones_count].steps = steps;
    strncpy(milestones[milestones_count].timestamp, timestamp, 31);
    milestones[milestones_count].timestamp[31] = '\0';
    milestones_count++;
}

static void write_milestones_json(void) {
    FILE *f = fopen(MILESTONES_JSON, "w");
    if (!f) return;
    fprintf(f, "[\n");
    for (size_t i = 0; i < milestones_count; i++) {
        fprintf(f, "    {\n");
        fprintf(f, "        \"number\": %llu,\n", (unsigned long long)milestones[i].number);
        fprintf(f, "        \"steps_milestone\": %d,\n", milestones[i].steps);
        fprintf(f, "        \"timestamp\": \"%s\"\n", milestones[i].timestamp);
        fprintf(f, "    }%s\n", (i + 1 < milestones_count) ? "," : "");
    }
    fprintf(f, "]\n");
    fclose(f);
}

static void get_timestamp(char *buf, size_t buflen) {
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    strftime(buf, buflen, "%Y-%m-%d %H:%M:%S", tm_info);
}

static void log_new_milestone(uint64_t number, int steps) {
    char ts[32];
    get_timestamp(ts, sizeof(ts));
    milestones_push(number, steps, ts);
    write_milestones_json();

    FILE *f = fopen(MILESTONES_TXT, "a");
    if (f) {
        fprintf(f, "[%s] Number: %llu | New Record Steps: %d\n", ts,
                (unsigned long long)number, steps);
        fclose(f);
    }
}

static void save_checkpoint(uint64_t next_seg_start, int overall_max_steps, uint64_t longest_number) {
    FILE *f = fopen(CHECKPOINT_FILE, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "    \"next_seg_start\": %llu,\n", (unsigned long long)next_seg_start);
    fprintf(f, "    \"overall_max_steps\": %d,\n", overall_max_steps);
    fprintf(f, "    \"longest_number\": %llu\n", (unsigned long long)longest_number);
    fprintf(f, "}\n");
    fclose(f);
}

// Minimal loader for the milestones JSON produced above (or by the Python version,
// which uses the same field names/shape) so a run can pick up the current record.
static void load_milestones(int *overall_max_steps, uint64_t *longest_number) {
    *overall_max_steps = 0;
    *longest_number = 0;

    FILE *f = fopen(MILESTONES_JSON, "r");
    if (!f) return;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return; }

    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    // Find the last occurrence of "number" and "steps_milestone" (crude but matches
    // the simple, flat JSON shape we write above).
    uint64_t last_number = 0;
    int last_steps = 0;
    char *p = buf;
    char *tag;
    while ((tag = strstr(p, "\"number\":")) != NULL) {
        last_number = strtoull(tag + strlen("\"number\":"), NULL, 10);
        p = tag + 1;
    }
    p = buf;
    while ((tag = strstr(p, "\"steps_milestone\":")) != NULL) {
        last_steps = (int)strtol(tag + strlen("\"steps_milestone\":"), NULL, 10);
        p = tag + 1;
    }
    free(buf);

    *overall_max_steps = last_steps;
    *longest_number = last_number;
}

// ---------------------------------------------------------
// 6. MAIN STREAMING ENGINE
// ---------------------------------------------------------
int main(void) {
    uint64_t overall_max_steps_u = 0;
    uint64_t longest_number = 0;
    int overall_max_steps = 0;

    load_milestones(&overall_max_steps, &longest_number);
    overall_max_steps_u = (uint64_t)overall_max_steps;
    (void)overall_max_steps_u;

    printf("=== STREAMING VERIFICATION ENGINE WITH MILESTONE TRACKING (C/OpenMP) ===\n");
    printf("Current Record Holder: %llu (%d steps)\n", (unsigned long long)longest_number, overall_max_steps);
    printf("OpenMP Threads Active: %d\n", omp_get_max_threads());

    struct timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    printf("Pre-generating base primes up to 1,000,000...\n");
    size_t n_base_primes;
    uint64_t *base_primes = get_base_primes(TOTAL_LIMIT, &n_base_primes);
    printf("Base primes count: %zu\n\n", n_base_primes);

    uint8_t *sieve_bytes = malloc((SEGMENT_SIZE / 8) + 1);
    int16_t *steps_out = malloc(SEGMENT_SIZE * sizeof(int16_t));

    uint64_t total_segments = TOTAL_LIMIT / SEGMENT_SIZE;

    printf("%-32s | %-10s | %-10s | %-10s | %s\n",
           "Segment Window", "Sieve (s)", "Test (s)", "Max Steps", "Progress");
    for (int i = 0; i < 85; i++) putchar('-');
    putchar('\n');

    bool counterexample_found = false;

    for (uint64_t seg_start = 0; seg_start < TOTAL_LIMIT && !counterexample_found; seg_start += SEGMENT_SIZE) {
        uint64_t seg_idx = seg_start / SEGMENT_SIZE;
        uint64_t seg_len = SEGMENT_SIZE;
        if (seg_start + seg_len > TOTAL_LIMIT) seg_len = TOTAL_LIMIT - seg_start;

        struct timespec t0, t1, t2;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        sieve_segment(sieve_bytes, seg_start, seg_len, base_primes, n_base_primes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double sieve_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        process_segment_parallel(seg_start, seg_len, sieve_bytes, steps_out);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double test_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;

        // Check counterexamples + milestones in index order (matches Python semantics)
        for (uint64_t i = 0; i < seg_len; i++) {
            int16_t val_steps = steps_out[i];
            if (val_steps < 0) {
                printf("\n[CRITICAL ALERT] Counterexample found at: %llu\n",
                       (unsigned long long)(seg_start + i));
                counterexample_found = true;
                break;
            }
            if (val_steps > overall_max_steps) {
                overall_max_steps = val_steps;
                longest_number = seg_start + i;
                printf("\n[MILESTONE] Number %llu hit %d steps!\n",
                       (unsigned long long)longest_number, overall_max_steps);
                log_new_milestone(longest_number, overall_max_steps);
            }
        }

        if (counterexample_found) break;

        save_checkpoint(seg_start + seg_len, overall_max_steps, longest_number);

        double pct = ((double)(seg_idx + 1) / (double)total_segments) * 100.0;
        char win_str[64];
        snprintf(win_str, sizeof(win_str), "[%llu - %llu]",
                 (unsigned long long)seg_start, (unsigned long long)(seg_start + seg_len - 1));

        printf("%-32s | %-10.3f | %-10.3f | %-10d | %.1f%%\n",
               win_str, sieve_time, test_time, overall_max_steps, pct);
    }

    for (int i = 0; i < 85; i++) putchar('-');
    putchar('\n');

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    double total_elapsed = (total_end.tv_sec - total_start.tv_sec) +
                            (total_end.tv_nsec - total_start.tv_nsec) / 1e9;

    printf("\nRUN COMPLETE!\n");
    printf("Total time elapsed: %.2f hours\n", total_elapsed / 3600.0);
    printf("Total Milestones Tracked: %zu\n", milestones_count);
    printf("Final Champion: %llu with %d steps\n",
           (unsigned long long)longest_number, overall_max_steps);

    free(base_primes);
    free(sieve_bytes);
    free(steps_out);
    free(milestones);
    return 0;
}