/*********************************************
 * OPL 22.1.0.0 Model
 * Author: 西南偏南
 * Creation Date: 2022年8月5日 at 下午8:01:38
 *********************************************/
dvar int+ x;
dvar int+ y;
minimize 2*x + 3 * y;
subject to{
  2 * x + 3 * y >= 20;
  x + y >= 0;
}
