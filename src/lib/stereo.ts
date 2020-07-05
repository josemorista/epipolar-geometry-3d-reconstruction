import { ndMat } from "../@types";

const exp2 = (x: number) => Math.pow(x, 2);

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


const compareWindowsBySSd = (w1: ndMat, w2: ndMat) => {
  let sum = 0;
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      sum += SSDError(w1[i][j], w2[i][j]);
    }
  }
  return sum;
};



const calculateWindowCorrelation = (w1: ndMat): number => {
  let averages = [0, 0, 0];
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      const v: any = w1[i][0];
      averages[0] += v[0];
      averages[1] += v[1];
      averages[2] += v[2];
    }
  }


  let variances = [0, 0, 0];
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      const v: any = w1[i][0];
      variances[0] += Math.pow((v[0] - averages[0]), 2);
      variances[1] += Math.pow((v[1] - averages[1]), 2);
      variances[2] += Math.pow((v[2] - averages[2]), 2);
    }
  }
  variances = variances.map(variance => (Math.sqrt(variance)));

  let resp = 0;
  for (let i = 0; i < w1.length; i++) {
    for (let j = 0; j < w1[i].length; j++) {
      const v: any = w1[i][0];
      resp += ((v[0] - averages[0]) / variances[0]) * ((v[1] - averages[1]) / variances[1]) * ((v[2] - averages[2]) / variances[2]);
    }
  }

  return resp;

};

const compareWindowsByCorrelation = (w1: ndMat, w2: ndMat) => {
  return Math.abs(calculateWindowCorrelation(w1) - calculateWindowCorrelation(w2));
};

export const searchMatchInEpiline = (img: ndMat, w: ndMat, row: number, errorFunction: 'SSD' | 'CORRELATION' = 'SSD') => {
  const errorsFunctions = {
    'SSD': compareWindowsBySSd,
    'CORRELATION': compareWindowsByCorrelation
  };
  let bestResult = {
    position: [] as Array<number>,
    error: Number.MAX_SAFE_INTEGER
  };
  for (let j = 0; j < img[row].length; j++) {
    const w2 = getPixelWindow(img, [row, j], w.length);
    if (w2) {
      const localError = errorsFunctions[errorFunction](w, w2);
      if (localError < bestResult.error) {
        bestResult.error = localError;
        bestResult.position = [row, j];
      }
    }
  }
  return bestResult.position;
};

