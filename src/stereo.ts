import * as cv from 'opencv4nodejs';
import path from 'path';
import { ndMat } from './@types';
import Matrix from 'ml-matrix';
import { matrix2ndMat } from './lib/utils';

const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im2.png'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im6.png'));

const img1 = img1cvMat.getDataAsArray();
const img2 = img2cvMat.getDataAsArray();

let disparity = matrix2ndMat(Matrix.zeros(img1.length, img1[0].length));

const exp2 = (x: number) => Math.pow(x, 2);

const SSDError = ([b1, g1, r1]: any, [b2, g2, r2]: any) => {
  return exp2(r2 - r1) + exp2(g2 - g1) + exp2(b2 - b1);
};

const getPixelWindow = (m: ndMat, [row, col]: Array<number>, windowSize = 15) => {
  let window = [] as ndMat;
  let wi = 0, wj = 0;

  let startRow = row - windowSize;
  let startCol = col - windowSize;

  let endRow = row + windowSize;
  let endCol = col + windowSize;

  if (startCol < 0 || startRow < 0 || endRow >= m.length || endCol >= m[0].length) {
    return null;
  }

  for (let i = startRow; i <= endRow; i++) {
    window.push([]);
    for (let j = startCol; j <= endCol; j++) {
      window[wi][wj] = m[i][j];
      wj++;
    }
    wj = 0;
    wi++;
  }

  return window;
};


// console.log(getPixelWindow([[1, 2, 3, 3], [4, 5, 6, 7], [7, 8, 9, 10], [11, 12, 13, 13]], [1, 2], 1));

const compareWindows = (w1: ndMat, w2: ndMat, errorFunction: (v1: any, v2: any) => number) => {
  let sum = 0;
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      sum += errorFunction(w1[i][j], w2[i][j]);
    }
  }
  return sum;
};


const searchMatchInEpiline = (img: ndMat, w: ndMat, row: number) => {
  let bestResult = {
    position: [] as Array<number>,
    error: Number.MAX_SAFE_INTEGER
  };
  for (let j = 0; j < img[row].length; j++) {
    const w2 = getPixelWindow(img, [row, j]);
    if (w2) {
      const localError = compareWindows(w, w2, SSDError);
      if (localError < bestResult.error) {
        bestResult.error = localError;
        bestResult.position = [row, j];
      }
    }
  }
  return bestResult.position;
};

const linearTransform = (x: number, min: number, max: number, a: number, b: number) => {
  return (b - a) * ((x - min) / (max - min)) + a;
};

let max = Number.MIN_SAFE_INTEGER, min = Number.MAX_SAFE_INTEGER;
const maxDisparity = 70;

for (let i = 0; i < img1.length; i++) {
  for (let j = 0; j < img1[i].length; j++) {
    const w = getPixelWindow(img1, [i, j]);
    if (w) {
      const matchPosition = searchMatchInEpiline(img2, w, i);
      let disp = Math.abs(j - matchPosition[1]);
      if (disp < maxDisparity) {
        if (disp < min) {
          min = disp;
        }
        if (disp > max) {
          max = disp;
        }
        disparity[i][j] = disp;
      }
    }
  }
  console.log(i);
}


let disparityRGB = [] as Array<Array<Array<number>>>;
console.log(min, max);

for (let i = 0; i < disparity.length; i++) {
  disparityRGB.push([]);
  for (let j = 0; j < disparity[i].length; j++) {
    const color = Math.round(linearTransform(disparity[i][j], min, max, 1, 255));
    disparityRGB[i][j] = [color, color, color];
  }
}

cv.imwrite('disparity.png', new cv.Mat(disparityRGB, cv.CV_8UC3));
