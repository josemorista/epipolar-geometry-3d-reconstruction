import { ndMat, IPoint } from "../../@types";
import Matrix, { SingularValueDecomposition } from "ml-matrix";
import * as cv from 'opencv4nodejs';

const exp2 = (x: number) => x * x;

const SSDError = ([b1, g1, r1]: any, [b2, g2, r2]: any) => {
  return exp2(r2 - r1) + exp2(g2 - g1) + exp2(b2 - b1);
};

export const getPixelWindow = (m: ndMat, [row, col]: Array<number>, windowSize = 3) => {
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


export const compareWindowsBySSd = (w1: ndMat, w2: ndMat) => {
  let sum = 0;
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      sum += SSDError(w1[i][j], w2[i][j]);
    }
  }
  return sum;
};

export const compareWindowsByCorrelation = (w1: ndMat, w2: ndMat) => {
  {
    const w1cv = new cv.Mat(w1, cv.CV_8UC3);
    const corr = w1cv.matchTemplate(new cv.Mat(w2, cv.CV_8UC3), cv.TM_CCOEFF_NORMED);
    return cv.minMaxLoc(corr).maxVal;
  }
};

export const computeEpilines = (pts: Array<IPoint>, F: ndMat, transpose: boolean = false) => {
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

export const calculateRightEpipole = (F: ndMat) => {
  const Fmat = (new Matrix(F)).transpose();
  const svdF = new SingularValueDecomposition(Fmat);
  return svdF.rightSingularVectors.getColumn(svdF.rightSingularVectors.columns - 1);
};


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
      if ((errorFunction === 'CORRELATION' && localError > bestResult.error) || (errorFunction === 'SSD' && localError < bestResult.error)) {
        bestResult.error = localError;
        bestResult.position = [i, j];
      }
    }
  }
  return bestResult.position;
};
