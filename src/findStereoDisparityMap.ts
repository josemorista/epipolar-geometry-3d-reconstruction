import * as cv from 'opencv4nodejs';
import path from 'path';
import Matrix from 'ml-matrix';
import { matrix2ndMat } from './lib/utils';
import { getPixelWindow, searchMatchInEpiline } from './lib/stereo';

const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im2.ppm'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im6.ppm'));

const img1 = img1cvMat.getDataAsArray();
const img2 = img2cvMat.getDataAsArray();

let disparity = matrix2ndMat(Matrix.zeros(img1.length, img1[0].length));


let max = Number.MIN_SAFE_INTEGER, min = Number.MAX_SAFE_INTEGER;
const maxDisparity = 64;

const linearTransform = (x: number, min: number, max: number, a: number, b: number) => {
  return (b - a) * ((x - min) / (max - min)) + a;
};

for (let i = 0; i < img1.length; i++) {
  for (let j = 0; j < img1[i].length; j++) {
    const w = getPixelWindow(img1, [i, j]);
    if (w) {
      const matchPosition = searchMatchInEpiline(img2, w, i, 'CORRELATION');
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
