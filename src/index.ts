import path from 'path';
import { siftMatches } from './lib/findMatches';
import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/eightPointAlgorithm';
import { ndMat, I2dPoint, IMatchPoint } from './@types';
import { matrix2ndMat } from './lib/common';

const calculateSquareError = (x: I2dPoint, xl: I2dPoint, F: ndMat) => {
  const xTmp = new Matrix([[x.x], [x.y], [1]]);
  let xlTmp = new Matrix([[xl.x, xl.y, 1]]);
  let M = xlTmp.mmul(new Matrix(F)).mmul(xTmp);
  let m = matrix2ndMat(M), sum = 0;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      sum += Math.pow(m[i][j], 2);
    }
  }
  return sum;
};

const getRandomInt = (min: number, max: number): number => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

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

const ransac = (samples: Array<IMatchPoint>, s: number, minError: number, k: number = 1, p: number = 0.95) => {
  const maxFrobenius = minError;
  let iterations = Number.MAX_SAFE_INTEGER;
  let bestSamples = {
    votes: -1,
    S: [] as Array<IMatchPoint>
  };
  while (iterations > k) {
    const { selectedSamples, selectedSamplesIndexes } = selectRandomSamples<IMatchPoint>(matches, s);
    let F = eightPointAlgorithm(selectedSamples);
    console.log(k);
    let Snow = matches.filter((el, index) => !selectedSamplesIndexes.includes(index)).filter(sample => {
      return calculateSquareError(sample.p1, sample.p2, F) <= maxFrobenius;
    });
    const actualVotes = Snow.length;
    if (actualVotes > bestSamples.votes) {
      bestSamples.votes = actualVotes;
      bestSamples.S = Snow;
      let e = 1 - (actualVotes / matches.length);
      iterations = Math.log(1 - p) / Math.log(1 - Math.pow((1 - e), s));
    }
    k++;
  }
  console.log(bestSamples.S.length);
  return eightPointAlgorithm(bestSamples.S);

};

const img1 = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2 = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const matches = siftMatches(img1, img2, 200);

const F = ransac(matches, 8, 0.01, 10000, 0.95);

// Display results
const cvMat = new cv.Mat(F, cv.CV_32F);
const { F: cv2Mat } = cv.findFundamentalMat(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);


const lines = cv.computeCorrespondEpilines(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), 2, cvMat);
const linesOpenCv = cv.computeCorrespondEpilines(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), 2, cv2Mat);

// Draw epipolar lines in BLUE
lines.forEach(r => {
  img2.drawLine(new cv.Point2(0, -r.z / r.y), new cv.Point2(img2.cols, -(r.z + r.x * img2.cols) / r.y), new cv.Vec3(255, 0, 0), 1);
});

// Draw opencv epipolar lines in RED
/*linesOpenCv.forEach(r => {
  img2.drawLine(new cv.Point2(0, -r.z / r.y), new cv.Point2(img2.cols, -(r.z + r.x * img2.cols) / r.y), new cv.Vec3(0, 0, 255), 1);
});*/

cv.imshowWait('result', img2);

let m = [[1, 1], [-1, -1]], sum = 0;
for (let i = 0; i < m.length; i++) {
  for (let j = 0; j < m[i].length; j++) {
    sum += Math.pow(m[i][j], 2);
  }
}
