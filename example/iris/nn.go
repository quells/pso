package main

import "math"

func feedForwardNN(weights []float64, input []float64) []float64 {
	// 4 inputs + bias; 4 nodes + bias; 4 nodes + bias; 3 outputs
	// 5*4 + 5*4 + 5*3 = 55 weights
	x := append(input, 1)
	l0 := math.Tanh(dot(weights[0:5], x))
	l1 := math.Tanh(dot(weights[5:10], x))
	l2 := math.Tanh(dot(weights[10:15], x))
	l3 := math.Tanh(dot(weights[15:20], x))
	l := []float64{l0, l1, l2, l3, 1.0}
	r0 := math.Tanh(dot(weights[20:25], l))
	r1 := math.Tanh(dot(weights[25:30], l))
	r2 := math.Tanh(dot(weights[30:35], l))
	r3 := math.Tanh(dot(weights[35:40], l))
	r := []float64{r0, r1, r2, r3, 1.0}
	o0 := math.Tanh(dot(weights[40:45], r))
	o1 := math.Tanh(dot(weights[45:50], r))
	o2 := math.Tanh(dot(weights[50:55], r))
	o := []float64{o0, o1, o2}
	return softmax(o)
}

func dot(u, v []float64) (result float64) {
	for i, ui := range u {
		result += ui * v[i]
	}
	return
}

func softmax(input []float64) []float64 {
	var sum float64
	e := make([]float64, len(input))
	for idx, i := range input {
		ei := math.Exp(i)
		e[idx] = ei
		sum += ei
	}
	for idx, ei := range e {
		e[idx] = ei / sum
	}
	return e
}

func argmax(input []float64) (idx int) {
	m := input[0]
	for i := 1; i < len(input); i++ {
		if input[i] > m {
			m = input[i]
			idx = i
		}
	}
	return
}
