import path from 'path';
import fs from 'fs';
import { siftMatches } from './lib/findMatches';
import { Matrix, SVD } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/eightPointAlgorithm';
import { ndMat, IMatchPoint, cvMat, IPoint } from './@types';
import { ransac } from './lib/ransac';
import { getRandomInt } from './lib/utils';
import { matrix2ndMat, readCameraProjectionMatrixFromFile } from './lib/utils';
import { passiveTriangulation } from './lib/passiveTriangulation';
import { compareWindowsBySSd, compareWindowsByCorrelation, getPixelWindow } from './lib/common';

const computeEpilines = (pts: Array<IPoint>, F: ndMat, transpose: boolean = false) => {
  let Fmat = new Matrix(F);
  if (transpose) {
    Fmat = Fmat.transpose();
  }
  let lines: Array<IPoint> = [];
  pts.forEach(p => {
    lines.push(Fmat.mmul(new Matrix([[...p, 1]]).transpose()).getColumn(0));
  });
  return lines;
};


const calculateRightEpipole = (F: ndMat) => {
  const Fmat = (new Matrix(F)).transpose();
  const svdF = new SVD(Fmat);
  return svdF.rightSingularVectors.getColumn(svdF.rightSingularVectors.columns - 1);
};

const drawLines = (lines: Array<IPoint>, img: cvMat) => {
  const c = img.cols;
  for (let i = 0; i < lines.length; i++) {
    const color = new cv.Vec3(getRandomInt(0, 255), getRandomInt(0, 255), getRandomInt(0, 255));
    let [x0, y0] = [0, -lines[i][2] / lines[i][1]].map(el => Math.round(el));
    let [x1, y1] = [c, -(lines[i][2] + lines[i][0] * c) / lines[i][1]].map(el => Math.round(el));
    img.drawLine(new cv.Point2(x0, y0), new cv.Point2(x1, y1), color, 1);
  }
};

const calculateSquareError = (sample: IMatchPoint, F: ndMat) => {
  const [x, y] = sample.p1, [xl, yl] = sample.p2;
  const xTmp = new Matrix([[x], [y], [1]]);
  let xlTmp = new Matrix([[xl, yl, 1]]);
  let M = xlTmp.mmul(new Matrix(F)).mmul(xTmp);
  let m = matrix2ndMat(M), sum = 0;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      sum += Math.pow(m[i][j], 2);
    }
  }
  return sum;
};

const img1cv = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2cv = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const matches = siftMatches(img1cv, img2cv, 1000);

let F = ransac(matches, 8, eightPointAlgorithm, calculateSquareError, 0.5, 0.95, 10);

const linesOnLeft = computeEpilines(matches.map(el => el.p1), F, true);
const linesOnRight = computeEpilines(matches.map(el => el.p1), F);

drawLines(linesOnRight, img2cv);
drawLines(linesOnLeft, img1cv);

cv.imshowWait('f1.png', img1cv);
cv.imshowWait('f2.png', img2cv);


// 3d reconstruction
const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const img1 = img1cvMat.getDataAsArray();
const img2 = img2cvMat.getDataAsArray();

const P1 = readCameraProjectionMatrixFromFile('temple0001.png');
const P2 = readCameraProjectionMatrixFromFile('temple0002.png');

let vertexList = [];

export const searchMatchInEpiline = (img: ndMat, w: ndMat, line: IPoint, errorFunction: 'SSD' | 'CORRELATION' = 'SSD') => {
  const errorsFunctions = {
    'SSD': compareWindowsBySSd,
    'CORRELATION': compareWindowsByCorrelation
  };
  let bestResult = {
    position: [] as Array<number>,
    error: Number.MAX_SAFE_INTEGER
  };
  for (let i = 0; i < img.length; i++) {
    const j = Math.round((line[1] * i + line[2]) / (-line[0]));
    const w2 = getPixelWindow(img, [i, j], w.length);
    if (w2) {
      const localError = errorsFunctions[errorFunction](w, w2);
      if (localError < bestResult.error) {
        bestResult.error = localError;
        bestResult.position = [i, j];
      }
    }
  }
  return bestResult.position;
};

for (let i = 0; i < img1.length; i++) {
  for (let j = 0; j < img1[i].length; j++) {
    const w = getPixelWindow(img1, [i, j], 3);
    if (w) {
      const epiline = computeEpilines([[i, j]], F)[0];
      const matchPosition = searchMatchInEpiline(img2, w, epiline, 'SSD');
      vertexList.push(passiveTriangulation([i, j], matchPosition, P1, P2));
    }
  }
  console.log(i);
}

fs.writeFileSync('vertexPoints.txt', JSON.stringify(vertexList));

// const { F: FcvMat } = cv.findFundamentalMat(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);