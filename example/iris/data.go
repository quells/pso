package main

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"strconv"
)

//go:embed iris.txt
var irisRaw []byte

type iris struct {
	values []float64
	label  int
}

var (
	trainingData []iris
	testingData  []iris
)

func init() {
	rows, err := csv.NewReader(bytes.NewReader(irisRaw)).ReadAll()
	if err != nil {
		panic(err)
	}

	for idx, row := range rows {
		parsed := iris{
			values: make([]float64, 4),
		}

		for i := 0; i < 4; i++ {
			parsed.values[i], _ = strconv.ParseFloat(row[i], 64)
		}

		switch row[4] {
		case "versicolor":
			parsed.label = 1
		case "virginica":
			parsed.label = 2
		}

		// could be randomized
		if idx%10 == 0 {
			testingData = append(testingData, parsed)
		} else {
			trainingData = append(trainingData, parsed)
		}
	}
}
