package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/quells/pso/pkg/swarm"
)

const (
	numEdges = 55
)

func main() {
	rand.Seed(time.Now().UnixNano())

	shape := make([]swarm.Range, numEdges)
	for i := 0; i < numEdges; i++ {
		shape[i][0] = -10.0
		shape[i][1] = 10.0
	}

	options := swarm.Options{
		PopulationSize: numEdges * 2,
		LocalSize:      2,
		WaitMagnitude:  2.5,
	}

	pso, err := swarm.New(train, shape, options)
	if err != nil {
		log.Fatalf("could not build swarm: %v", err)
	}

	pso.StepUntil(1e-6)

	fmt.Println(test(pso.Best()))
	// fmt.Println(pso.Best())
}

func train(weights []float64) (score float64) {
	for _, flower := range trainingData {
		predicted := feedForwardNN(weights, flower.values)[flower.label]
		score -= predicted
	}
	return
}

func test(weights []float64) (score float64) {
	for _, flower := range trainingData {
		predicted := argmax(feedForwardNN(weights, flower.values))
		if predicted == flower.label {
			score++
		}
	}
	score /= float64(len(trainingData))
	return
}
