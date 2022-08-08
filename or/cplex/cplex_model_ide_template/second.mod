/*********************************************
 * OPL 22.1.0.0 Model
 * Author: ����ƫ��
 * Creation Date: 2022��8��5�� at ����8:22:28
 *********************************************/
int n =4;
int C = 13;
int p[1..n] = [12, 11, 9, 8];
int w[1..n] = [8,6,4,3];

dvar boolean x[1..n];

maximize sum(j in 1..n) p[j] * x[j];

subject to {
  sum(j in 1..n) w[j] * x[j] <= C;
  }