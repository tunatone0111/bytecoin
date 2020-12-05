from .mongodb import get_db

db = get_db()
cache = db['StocksCache']
cache.drop()

pipeline = [{
  '$lookup': {
    'from': 'Posts',
    'let': {
      'c': '$code'
      },
    'pipeline': [{
      '$match': {
        '$expr': {'$eq': ['$code','$$c']}
      }
      },{
      '$group': {
        '_id': None,
        'count': {'$sum': 1},
        'label': {'$avg': '$label'}
      }}],
      'as': 'posts'
      }},{
  '$match': {'posts': {'$ne': []}}
  },{
      '$project': {
        'name': 1,
        'code': 1,
        'label': {'$first': '$posts'}
        }}, {'$project': {
  'name': 1,
  'code': 1,
  'label': '$label.label',
  'numPosts': '$label.count'
}}, {'$out': 'StocksCache'}]

stocks = db['Stocks']
stocks.aggregate(pipeline)