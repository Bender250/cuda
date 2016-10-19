// naive CPU implementation

void solveCPU(const int *results, float *avg_stud, float*avg_que, const int students, const int questions){
    for (int s = 0; s < students; s++) {
        int stud = 0;
        for (int q = 0; q < questions; q++) {
            stud += results[s*questions + q];
        }
        avg_stud[s] = (float)stud / (float)questions;
    }
    for (int q = 0; q < questions; q++) {
        int que = 0;
        for (int s = 0; s < students; s++) {
            que += results[s*questions + q];
        }
        avg_que[q] = (float)que / (float)students;
    }
}

