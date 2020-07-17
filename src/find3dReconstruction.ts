import path from 'path';
import fs from 'fs';
import { siftMatches } from './lib/3dReconstruction/findMatches';
import { Matrix } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/3dReconstruction/eightPointAlgorithm';
import { ndMat, IMatchPoint, cvMat, IPoint } from './@types';
import { ransac } from './lib/Optimization/ransac';
import { getRandomInt, normalizeVertexList } from './lib/utils';
import { matrix2ndMat, readCameraProjectionMatrixFromFile } from './lib/utils';
import { passiveTriangulation } from './lib/3dReconstruction/passiveTriangulation';
import { computeEpilines, searchMatchInEpiline, getPixelWindow } from './lib/3dReconstruction/epilines';

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
  // return M.norm("frobenius");
  let m = matrix2ndMat(M), sum = 0;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      sum += (m[i][j] * m[i][j]);
    }
  }
  return sum;
};

const img1cv = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2cv = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const matches = siftMatches(img1cv, img2cv, 200);

let F = ransac(matches, 8, eightPointAlgorithm, calculateSquareError, 0.01, 0.95, 10);

// const linesOnLeft = computeEpilines(matches.map(el => el.p2), F, true);
const linesOnRight = computeEpilines(matches.map(el => el.p1), F);

drawLines(linesOnRight, img2cv);
// drawLines(linesOnLeft, img1cv);

cv.imshowWait('f2.png', img2cv);


// 3d reconstruction
const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const img1 = img1cvMat.getDataAsArray();
const img2 = img2cvMat.getDataAsArray();

const P1 = readCameraProjectionMatrixFromFile('temple0001.png');
const P2 = readCameraProjectionMatrixFromFile('temple0002.png');

let vertexList = [];
for (let i = 0; i < img1.length; i++) {
  for (let j = 0; j < img1[i].length; j++) {
    const w = getPixelWindow(img1, [i, j], 5);
    if (w) {
      const epiline = computeEpilines([[i, j]], F)[0];
      const matchPosition = searchMatchInEpiline(img2, w, epiline, 'SSD');
      vertexList.push(passiveTriangulation([i, j], matchPosition, P1, P2));
    }
  }
  console.log(i);
}

// fs.writeFileSync('vertexPoints.txt', JSON.stringify(normalizeVertexList(vertexList)));
fs.writeFileSync('vertexPoints.txt', JSON.stringify(vertexList));
