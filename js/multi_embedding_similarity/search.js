const info = require("debug")("info");
const debug = require("debug")("debug");

const { MongoClient } = require('mongodb');

const config = require("./config.json")
const fnames = require("./data/fnames.json");

const TENANT_NAME = config.mongo.tenantId;
const LIMIT=10;

async function search(vector, vectorIndex, limit, tenant) {
    const client = new MongoClient(config.mongo.connectionString);

    const database = client.db(config.mongo.db);
    const collection = database.collection(config.mongo.collection);

    const agg = [
      {
        '$vectorSearch': {
          'index': config.mongo.vectorIndex,
          'path': config.mongo.vectorFieldName,
          'queryVector': vector,
          'numCandidates': 150,
          'limit': limit,
          'filter': { [config.mongo.tenantIdFieldName]: tenant } // additional filtering by tenancy
        }
      }, {
        '$project': {
          '_id': 0,
          'aimId': '$_id',
          'alias': 1,
          'vectorIndex': 1,
          [config.mongo.tenantIdFieldName]: 1,
          [config.mongo.vectorFieldName]: 1,
          'distance': {
            '$meta': 'vectorSearchScore'
          }
        }
      }
    ];      

    const aggregateResult = collection.aggregate(agg);
    
    let result = [];

    await aggregateResult.forEach((item) => {
      result.push({
        distance:   item.distance,
        alias:      item.alias,
        deltaIndex: item.vectorIndex - vectorIndex
        // tenant:   item[config.mongo.tenantIdFieldName]
      })      
    });

    return result;
}

function w(x) { return 1/(1 + x*x); }

function compareResults(ra, rb) {
  if (ra.distance > rb.distance) {
    return -1;
  } else if (ra.distance < rb.distance) {
    return 1;
  }
  return 0;
}

function isInArray(arr, audioName) {
  return arr.includes(audioName);
}

function getAllAudioNames(results) {
  let names = []
  for (let fragmentSearchResults of results) {
    for (let result of fragmentSearchResults) {
      if (!isInArray(names, result.alias)) {
        names.push(result.alias);
      }
    }
  }

  debug("allNames: %o", names);
  return names;
}

function getAudioDistance(walker, dimensions, results, audioName) {
  let distanceM = 0;
  let deltaIndexM = 0;
  let K = 0;

  for (let i=0; i<walker.length; i++) {
    if (dimensions[i].length > 0) { // есть данные
      let res = results[i][dimensions[i][walker[i]]]

      distanceM = distanceM + res.distance;
      deltaIndexM = deltaIndexM + res.deltaIndex;
    } else {
      K = K + 1; // считаем сколько раз песня не попала в выборку вобще
    }
  } 

  distanceM = distanceM / walker.length;
  deltaIndexM = deltaIndexM / walker.length;

  let deltaIndexD = 0

  for (let i=0; i<walker.length; i++) {
    if (dimensions[i].length > 0) { // есть данные
      let res = results[i][dimensions[i][walker[i]]]
      deltaIndexD = deltaIndexD + (res.deltaIndex - deltaIndexM)*(res.deltaIndex - deltaIndexM);
    }
  } 
  
  deltaIndexD = deltaIndexD / walker.length;

  let result = {
    alias: audioName,
    //distance: distanceM * w(K) * w(deltaIndexD),
    distance: distanceM * w(deltaIndexD),
  }

  return result;
}

function incWalker(walker, dimensions) {
  let newWalker = [];
  let crop = true;

  for (let i=0; i<walker.length; i++) {
    let r = walker[i];
    
    if (crop) {
      r++;      
      crop = false;

      if (r >= dimensions[i].length) {
        r = 0;
        crop = true;
      }
    }

    newWalker.push(r);
  }

  return newWalker;
}

function isZeroWalker(walker) {
  for (let item of walker) {
    if (item != 0) {
      return false;
    }
  }

  return true;
}

function getBestResultForAudio(audioName, results) {
  debug("getBestResultForAudio: %s", audioName)
  let walker = [];
  let dimensions = []; //массив из массива индексов audioName в results

  let deltaDims = [];

  for (let i=0; i<results.length; i++) {

    let indexes = [];
    let deltas = [];

    for (let j=0; j<results[i].length; j++) {
      if (results[i][j].alias == audioName) {
        indexes.push(j);

        deltas.push(results[i][j].deltaIndex);
      }
    }

    dimensions.push(indexes);
    deltaDims.push(deltas);

    walker.push(0);
  }

  debug("dimensions: %o", dimensions);
  debug("deltas    : %o", deltaDims);

  let bestAudioDistance = getAudioDistance(walker, dimensions, results, audioName);
  walker = incWalker(walker, dimensions);

  while (!isZeroWalker(walker)) {
    let audioDistance = getAudioDistance(walker, dimensions, results, audioName);
    if (audioDistance.distance > bestAudioDistance.distance) {
      bestAudioDistance = audioDistance;
    }

    walker = incWalker(walker, dimensions);
  }

  return bestAudioDistance;
}

function filterResult(searchResults) {
  let filtered = [];
  filtered.push(searchResults[0]);
  for (let i=1; i<searchResults.length; i++) {

    let isAppendNeeded = true;
    for (let j=0; j<filtered.length; j++) {

      if (filtered[j].alias == searchResults[i].alias) {
        if (searchResults[i].distance > filtered[j].distance) {
          filtered[j].distance = searchResults[i].distance;
          filtered[j].deltaIndex = searchResults[i].deltaIndex;
        }
        isAppendNeeded = false;
        break;
      }
    }

    if (isAppendNeeded) {
      filtered.push(searchResults[i]);
    }
  }

  filtered.sort(compareResults);
  return filtered;
}

function guessResultsForAudio(oldResults, originalAudioName) {
  let results = []
  for (let i=0; i<oldResults.length; i++) {
    results.push(filterResult(oldResults[i]));
  }

  let songResults = [];
  let audioNames = getAllAudioNames(results);
  for (let audioName of audioNames) {
    let songResult = getBestResultForAudio(audioName, results);
    songResults.push(songResult);
  }
  
  songResults.sort(compareResults);

  return {isGuess: songResults[0].alias == originalAudioName, variants: songResults};
}

async function main() {
  info("Started...");

  let guessCount = 0;
  let failCount = 0;

  for (let f of fnames) {
    if (f.type === "diffusion") {
      const embeddings = require(`./data/embeddings/${f.name}.json`);

      let searchResults = []
      for (let i=0; i < embeddings.audio.embeddings.length; i++) {
        let r = await search(embeddings.audio.embeddings[i], i, LIMIT, TENANT_NAME);
        searchResults.push(r);
        // info("Song(%s): %o", embeddings.audio.name, r);
      }

      let guess = guessResultsForAudio(searchResults, f.original);
      info("song: %s -> %o", f.name, guess);      
      if (guess.isGuess) {
        guessCount++;
      } else {
        failCount++;
      }
    }
  }

  info("Total guess:fail = %i:%i", guessCount, failCount);
}

main().then((res, rej) => {console.log(res)}).catch(console.dir);


