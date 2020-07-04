import * as cv from 'opencv4nodejs';
import { cvMat, IMatchPoint } from '../@types';

const drawMatches = false;

const matchFeatures = (
  img1: cvMat,
  img2: cvMat,
  detector: cv.FeatureDetector,
  matchFunc: (descs1: cvMat, descs2: cvMat) => cv.DescriptorMatch[],
  bestN = 8
) => {
  // detect keypoints
  const keyPoints1 = detector.detect(img1);
  const keyPoints2 = detector.detect(img2);

  // compute feature descriptors
  const descriptors1 = detector.compute(img1, keyPoints1);
  const descriptors2 = detector.compute(img2, keyPoints2);

  // match the feature descriptors
  const matches = matchFunc(descriptors1, descriptors2);

  // only keep good matches
  const bestMatches = matches.sort(
    (match1, match2) => match1.distance - match2.distance
  ).slice(0, bestN);

  // create an array with img coordinates for the matches
  let matchedPoints: Array<IMatchPoint> = [];
  bestMatches.forEach(match => {
    matchedPoints.push({
      p1: { x: keyPoints1[match.queryIdx].pt.x, y: keyPoints1[match.queryIdx].pt.y },
      p2: { x: keyPoints2[match.trainIdx].pt.x, y: keyPoints2[match.trainIdx].pt.y }
    });
  });

  // draw matches to debug
  if (drawMatches) {
    const matchedImg = cv.drawMatches(
      img1,
      img2,
      keyPoints1,
      keyPoints2,
      bestMatches
    );
    cv.imshowWait('Matched img', matchedImg);
  }

  // return matchedPoints for both images
  return matchedPoints;

};

export const siftMatches = (img1: cvMat, img2: cvMat, numberOfMatches = 40) => {
  return matchFeatures(
    img1,
    img2,
    new cv.SIFTDetector({ nFeatures: 2000 }),
    cv.matchFlannBased,
    numberOfMatches
  );
};

export const orbMatches = (img1: cvMat, img2: cvMat, numberOfMatches = 40) => {
  return matchFeatures(
    img1,
    img2,
    new cv.ORBDetector(),
    cv.matchBruteForceHamming,
    numberOfMatches
  );
};