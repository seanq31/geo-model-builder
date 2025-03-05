(param O point)
(param A point)
(define Gamma circle (coa O A))
(assert (= (dist O A) 5))      ;; 设定圆半径为5
(param B point (on-circ Gamma)) ;; B在圆上
(define M point (midp A B))     ;; AB中点
(assert (= (dist O M) 3))       ;; 中点距圆心3
(eval (= (dist A B) 8))         ;; 验证弦长是否为8