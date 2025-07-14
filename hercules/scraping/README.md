```bash
docker run -it -d \
  --name nethackwiki \
  -p 8080:80 \
  -v "$(pwd)":/var/www/html/dumps:ro \
  -v "$(pwd)/data":/var/www/html/data \
  -e MEDIAWIKI_DB_TYPE=sqlite \
  -e MEDIAWIKI_DB_NAME=nethackwiki \
  mediawiki:1.39
```


```bash
docker run -it -d \
  --name nethackwiki \
  -p 8080:80 \
  -v "$(pwd)":/var/www/html/dumps:ro \
  -v "$(pwd)/data":/var/www/html/data \
  -v "$(pwd)/extensions":/var/www/html/extensions:ro \
  -v "$(pwd)/data/LocalSettings.php":/var/www/html/LocalSettings.php \
  -v "$(pwd)/sqlite":/var/www/data \
  -e MEDIAWIKI_DB_TYPE=sqlite \
  -e MEDIAWIKI_DB_NAME=/var/www/data/nethackwiki.sqlite \
  mediawiki:1.39
```
