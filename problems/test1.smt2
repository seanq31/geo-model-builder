(param A point)
(param B point)
(param C point)

(assert (= (dist A B) 10))
(assert (= (dist A C) 10))
(assert (= (dist B C) 10))

(param L1 line (line A B))