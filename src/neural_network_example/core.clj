(ns neural-network-example.core
  (:require [clojure.core.matrix :as matrix])
  (:gen-class))

;;Neurons
;;  Input Hidden  Output
;;  A     1       C
;;  B     2       D
;;        3


;; Connection Strengths
;; Input to Hidden => [[A1 A2 A3] [B1 B2 B3]]
;; Hidden to Output => [[1C 1D] [2C 2D] [3C 3D]]

(def learning-rate 0.2)
(def input-neurons [1 0])
(def input-hidden-strengths [[0.12 0.2 0.13]
                             [0.01 0.02 0.03]])
(def hidden-neurons [0 0 0])
(def hidden-output-strengths [[0.15 0.16]
                              [0.02 0.03]
                              [0.01 0.02]])

;; -------------------
;; Feed Forward
;; -------------------
(def activation-fn (fn [x] (Math/tanh x)))
(def dactivation-fn (fn [y] (- 1.0 (* y y))))

(defn layer-activation
  "Forward propagate the input of a layer"
  [inputs strengths]
  (mapv activation-fn
        (mapv #(reduce + %)
              (matrix/mul inputs (matrix/transpose strengths)))))

;; Remember the new-hidden-neurons
(def new-hidden-neurons
  (layer-activation input-neurons input-hidden-strengths))

;; Remember the new-output-neurons
(def new-output-neurons
  (layer-activation new-hidden-neurons hidden-output-strengths))

;; -------------------
;; Backwards Propagation
;; -------------------

;; To train our network, we have to let it know what the answer,(or target), should be,
;; so we can calculate the errors and finally update our connection strengths.
;; For this simple example, let’s just inverse the data – so given an input of
;; [1 0] should give us an output of [0 1].

(def targets [0 1])

;; Calculate the error of the output layer
(defn output-deltas
  "Measures the delta errors for the output layer (Desired value – actual value)
  and multiplying it by the gradient of the activation function"
  [targets outputs]
  (matrix/mmul (mapv dactivation-fn outputs)
               (matrix/sub targets outputs)))

;; Remember the output deltas
(def odeltas (output-deltas targets new-output-neurons))

;; Calculate the error of the hidden layer
(defn hlayer-deltas
  [odeltas neurons strengths]
  (matrix/mmul (mapv dactivation-fn neurons)
               (mapv #(reduce + %)
                     (matrix/mul odeltas strengths))))

;; Remember the hiddent deltas
(def hdeltas (hlayer-deltas
              odeltas
              new-hidden-neurons
              hidden-output-strengths))

;; -------------------
;; Updating the connection strength
;; -------------------

;; weight-change = error-delta * neuron-value
;; new-weight = weight + learning rate * weight-change

(defn update-strengths
  [deltas neurons strengths lrate]
  (matrix/transpose
    (matrix/add
      (matrix/transpose strengths)
      (matrix/mul lrate (mapv #(* deltas %) neurons)))))

;; Update the hidden-output strengths
;; weight-change = odelta * hidden value
;; new-weight = weight + (learning rate * weight-change)

(def new-hidden-output-strengths
  (update-strengths
    odeltas
    new-hidden-neurons
    hidden-output-strengths
    learning-rate))

;; -------------------
;; Updating the input-hidden strength
;; -------------------

;; weight-change = hdelta * input-value
;; new-weight = weight + learning rate * weight-change

(def new-input-hidden-strengths
  (update-strengths
    hdeltas
    input-neurons
    input-hidden-strengths
    learning-rate))

;; Now putting the pieces together
;; 1. Forward propagated the input to get the output
;; 2. Calculated the errors from the target through backpropogation
;; 3. Updated the connection strengths/ weights

;; Construct a network representation
;; We will start off with all the values of the neurons being zero.
(def nn [[0 0]
         input-hidden-strengths
         hidden-neurons
         hidden-output-strengths
         [0 0]])

;; Generalized feed forward
(defn feed-forward
  [input network]
  (let [[in i-h-strengths h h-o-strengths out] network
        new-h (layer-activation input i-h-strengths)
        new-o (layer-activation new-h h-o-strengths)]
    [input i-h-strengths new-h h-o-strengths new-o]))

;; Generalized update weights / connection strengths
(defn update-weights
  [network target learning-rate]
  (let [[in i-h-strengths h h-o-strengths out] network
        o-deltas (output-deltas target out)
        h-deltas (hlayer-deltas o-deltas h h-o-strengths)
        n-h-o-strengths (update-strengths
                          o-deltas
                          h
                          h-o-strengths
                          learning-rate)
        n-i-h-strengths (update-strengths
                          h-deltas
                          in
                          i-h-strengths
                          learning-rate)]
    [in n-i-h-strengths h n-h-o-strengths out]))

;; Generalized train network
(defn train-network
  [network input target learning-rate]
  (update-weights (feed-forward input network) target learning-rate))

(def n1 (-> nn
            (train-network [1 0] [0 1] 0.5)
            (train-network [0.5 0] [0 0.5] 0.5)
            (train-network [0.25 0] [0 0.25] 0.5)))

;; It would be nice to have a training data structure look like:
;; [ [input target] [input target] ...]
(defn train-data
  [network data learning-rate]
  (if-let [[input target] (first data)]
    (recur
      (train-network network input target learning-rate)
      (rest data)
      learning-rate)
    network))

(def n2 (train-data nn [[[1 0] [0 1]]
                        [[0.5 0] [0 0.5]]
                        [[0.25 0] [0 0.25]]]
                    0.5))

(defn inverse-data
  []
  (let [n (rand 1)]
    [[n 0] [0 n]]))

(def n3 (train-data nn (repeatedly 400 inverse-data) 0.5))

;; General Construct Network
(defn gen-strengths
  [to from]
  (let [l (* to from)]
    (map vec (partition from (repeatedly l #(rand (/ 1 l)))))))

(defn construct-network
  [num-in num-hidden num-out]
  (vec (map vec [(repeat num-in 0)
                 (gen-strengths num-in num-hidden)
                 (repeat num-hidden 0)
                 (gen-strengths num-hidden num-out)
                 (repeat num-out 0)])))

(def tnn (construct-network 2 3 2))
(def n5 (train-data tnn (repeatedly 1000 inverse-data) 0.2))
