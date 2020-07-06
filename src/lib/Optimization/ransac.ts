import { getRandomInt } from "../utils";

const selectRandomSamples = <T>(samples: Array<T>, numberOfSelect: number) => {
  let selected: Array<T> = [];
  let selectedIndexes: Array<number> = [];
  while (selected.length < numberOfSelect) {
    let i = getRandomInt(0, samples.length - 1);
    if (!selectedIndexes.includes(i)) {
      selected.push(samples[i]);
      selectedIndexes.push(i);
    }
  }
  return { selectedSamples: selected, selectedSamplesIndexes: selectedIndexes };
};

export const ransac = <T>(samples: Array<T>, s: number, functionToApply: (s: Array<T>) => any, errorFunction: (s: T, f: any) => number, minError: number, p: number = 0.95, minIterations = 1) => {
  let iterations = Number.MAX_SAFE_INTEGER;
  let bestSamples = {
    votes: -1,
    S: [] as Array<T>,
  };
  let k = 1;
  while (iterations > k || k < minIterations) {
    const { selectedSamples, selectedSamplesIndexes } = selectRandomSamples<T>(samples, s);
    let F = functionToApply(selectedSamples);
    let Snow = samples.filter((el, index) => !selectedSamplesIndexes.includes(index)).filter(sample => {
      return (errorFunction(sample, F) < minError);
    });
    const actualVotes = Snow.length;
    if (actualVotes > bestSamples.votes) {
      bestSamples.votes = actualVotes;
      bestSamples.S = [...Snow];
      let e = 1 - (actualVotes / samples.length);
      iterations = Math.log(1 - p) / Math.log(1 - Math.pow((1 - e), s));
    }
    k++;
  }
  console.log(`${k} loops - ${bestSamples.S.length} votes!`);
  if (bestSamples.S.length < s) {
    console.log('Ransac error, raise your minValue parameter!');
    return [[]];
  }

  return functionToApply(bestSamples.S);

};
