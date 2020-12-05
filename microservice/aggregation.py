from mongodb import get_db
import datetime
from pytz import timezone

ctime = datetime.datetime.now() + datetime.timedelta(hours=9)
ttime = ctime - datetime.timedelta(days=1)

pipeline = [{
  '$lookup': {
    'from': 'Posts',
    'let': {
      'c': '$code'
      },
    'pipeline': [{
      '$match': {
        '$expr': {'$and': [
          {'$eq': ['$code','$$c']},
          {'$gt': ['$date', ttime]}
        ]}
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

db = get_db()
db['Stocks'].aggregate(pipeline)

print(f'[SUCCESS] aggregation complete, ${ctime} to ${ttime}')