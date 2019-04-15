from optparse import OptionParser
import re

BROADER_STR = '<http://www.w3.org/2004/02/skos/core#broader>'
SUBJECT_STR = '<http://www.w3.org/2004/02/skos/core#subject>'

def build_category_map(dir):
    result = {}
    with open(dir + '/skoscategories_en.nt', 'r') as f:
        for line in f:
            fields = line.split(' ')
            sub_cat = fields[0]
            rel = fields[1]
            cat = fields[2]
            if rel == BROADER_STR:
                result[sub_cat] = cat
    return result

def build_article_map(dir):
    result = {}
    with open(dir + '/articlecategories_en.nt', 'r') as f:
        for line in f:
            fields = line.split(' ')
            article_url = fields[0]
            rel = fields[1]
            article_cat = fields[2]
            if rel == SUBJECT_STR:
                result[article_url] = article_cat
    return result

def append_super_cat(dir, cat_map, article_map):
    with open(dir + '/longabstract_en.nt', 'r') as f:
        with open(dir + '/joinedlonabstract_en.nt', 'w') as g:
            for line in f:
                match = re.search(r'(^<http://dbpedia.org/[^<]+)\s', line)
                article_url = match.group(1)
                article_cat = article_map.get(article_url, None)
                if not article_cat:
                    print('Warning: %s has no category!' % article_url)
                    continue
                article_super_cat = cat_map.get(article_cat, None)
                if not article_super_cat:
                    print('Warning: %s has no super category!' % article_url)
                    article_super_cat = 'None'
                g.write('%s %s %s\n' % (line.strip(), article_cat, article_super_cat))

def main():
    parser = OptionParser()
    parser.add_option('-d', '--dir', dest='dir',
                      help='path to directory holding .nt data files')

    (option, args) = parser.parse_args()

    if not option.dir:
        print('please specify -d <directory>')
        sys.exit(1)

    cat_map = build_category_map(option.dir)
    article_map = build_article_map(option.dir)
    append_super_cat(option.dir, cat_map, article_map)
    print('done')
