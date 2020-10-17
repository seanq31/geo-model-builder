(param (A B C D) polygon)
(assert (right A B C))
(assert (right C D A))
(compute H point (foot A (line B D)))
(param S point (onSeg A B))
(param T point (onSeg A D))
(assert (inPoly H S C T))
(assert (= (div pi 2) (sub (uangle C H S) (uangle C S B))))
(assert (= (div pi 2) (sub (uangle T H C) (uangle D T C))))

(confirm (tangentLC (line B D) (circumcircle T S H)))