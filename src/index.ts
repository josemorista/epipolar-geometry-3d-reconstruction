import path from 'path';
import { siftMatches } from './lib/findMatches';
import { Matrix } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/eightPointAlgorithm';
import { ndMat, I2dPoint, IMatchPoint, cvMat } from './@types';
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

const ransac = (samples: Array<IMatchPoint>, s: number, minError: number, p: number = 0.95) => {
  let iterations = Number.MAX_SAFE_INTEGER;
  let bestSamples = {
    votes: -1,
    S: [] as Array<IMatchPoint>,
  };
  let k = 1;
  while (iterations > k) {
    const { selectedSamples, selectedSamplesIndexes } = selectRandomSamples<IMatchPoint>(samples, s);
    let F = eightPointAlgorithm(selectedSamples);
    let Snow = matches.filter((el, index) => !selectedSamplesIndexes.includes(index)).filter(sample => {
      return (calculateSquareError(sample.p1, sample.p2, F) < minError);
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
  console.log(`${k} loops - ${bestSamples.S.length} votes!`);
  if (bestSamples.S.length < s) {
    console.log('Ransac error, raise your minValue parameter!');
    return [[]];
  }
  return eightPointAlgorithm(bestSamples.S);

};

const img1 = cv.imread(path.resolve('.', 'datasets', 'umbrella', 'im0.png'));
const img2 = cv.imread(path.resolve('.', 'datasets', 'umbrella', 'im1.png'));

const matches = siftMatches(img1, img2, 10000);

const F = ransac(matches, 9, 0.01, 0.9);

const FcvMat = new cv.Mat(F, cv.CV_32F);
// const { F: FcvMat } = cv.findFundamentalMat(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);

const pts1 = matches.map(el => new cv.Point2(el.p1.x, el.p1.y));
const pts2 = matches.map(el => new cv.Point2(el.p2.x, el.p2.y));

const linesOnLeft = cv.computeCorrespondEpilines(pts2, 2, FcvMat);
const linesOnRight = cv.computeCorrespondEpilines(pts1, 1, FcvMat);

const drawLines = (lines: Array<cv.Vec3>, img1: cvMat, img2: cvMat, pts1: Array<cv.Point2>, pts2: Array<cv.Point2>) => {
  const c = img1.cols;
  for (let i = 0; i < lines.length; i++) {
    const color = new cv.Vec3(getRandomInt(0, 255), getRandomInt(0, 255), getRandomInt(0, 255));
    let [x0, y0] = [0, -lines[i].z / lines[i].y].map(el => Math.round(el));
    let [x1, y1] = [c, -(lines[i].z + lines[i].x * c) / lines[i].y].map(el => Math.round(el));
    img1.drawLine(new cv.Point2(x0, y0), new cv.Point2(x1, y1), color, 1);
    img1.drawCircle(new cv.Point2(pts1[i].x, pts2[i].y), 5, color, -1);
    img2.drawCircle(new cv.Point2(pts2[i].x, pts2[i].y), 5, color, -1);
  }
};

drawLines(linesOnLeft, img1, img2, pts1, pts2);
drawLines(linesOnRight, img2, img1, pts2, pts1);

cv.imwrite('result1.png', img1);
cv.imwrite('result2.png', img2);
