import fs from 'fs';
import path from 'path';
import { ndMat } from './@types';
import * as cv from 'opencv4nodejs';
import Matrix from 'ml-matrix';
import { matrix2ndMat } from './lib/utils';
import { getPixelWindow, searchMatchInEpiline } from './lib/stereo';
import { passiveTriangulation } from './lib/passiveTriangulation';

const fileLines = fs.readFileSync(path.resolve('.', 'datasets', 'temple', 'temple_par.txt'), 'utf-8').split('\n');


const readCameraProjectionMatrixFromFile = (filename: string) => {
  let K = [] as ndMat;
  let R = [] as ndMat;

  fileLines.forEach(line => {
    const parts = line.split(' ');
    if (parts[0] === filename) {
      for (let i = 0; i < 3; i++) {
        K[i] = [];
        for (let j = 0; j < 3; j++) {
          K[i][j] = Number(parts[(i * 3 + j) + 1]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i] = [];
        for (let j = 0; j < 3; j++) {
          R[i][j] = Number(parts[(i * 3 + j) + 10]);
        }
      }
      for (let i = 0; i < 3; i++) {
        R[i].push(Number(parts[i + 19]));
      }
    }
  });

  return matrix2ndMat((new Matrix(K)).mmul(new Matrix(R)));
};

const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));


const img1 = img1cvMat.getDataAsArray();
const img2 = img2cvMat.getDataAsArray();


const P1 = readCameraProjectionMatrixFromFile('temple0001.png');
const P2 = readCameraProjectionMatrixFromFile('temple0002.png');

let vertex = [];
const maxDisparity = 255;

for (let i = 0; i < img1.length; i++) {
  for (let j = 0; j < img1[i].length; j++) {
    const w = getPixelWindow(img1, [i, j], 5);
    if (w) {
      const matchPosition = searchMatchInEpiline(img2, w, i, 'SSD');
      let disp = Math.abs(j - matchPosition[1]);
      if (disp < maxDisparity) {
        vertex.push(passiveTriangulation([i, j], matchPosition, P1, P2));
      }
    }
  }
  console.log(i);
}

fs.writeFileSync('vertexPoints.txt', JSON.stringify(vertex));