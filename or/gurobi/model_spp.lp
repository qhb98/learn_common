\ Model dual problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5 x_s,a + 8 x_s,b + 2 x_a,c - 10 x_b,a + 3 x_c,b + 4 x_b,t + 3 x_c,t
Subject To
 start_flow: x_s,a + x_s,b = 1
 end_flow: x_b,t + x_c,t = 1
 flow_conservation: = 0
 flow_conservation: x_s,a - x_a,c + x_b,a = 0
 flow_conservation: x_s,b - x_b,a + x_c,b - x_b,t = 0
 flow_conservation: x_a,c - x_c,b - x_c,t = 0
 flow_conservation: = 0
Bounds
Binaries
 x_s,a x_s,b x_a,c x_b,a x_c,b x_b,t x_c,t
End
