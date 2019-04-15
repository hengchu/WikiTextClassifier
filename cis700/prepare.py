from collections import Counter
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
            if rel == BROADER_STR and sub_cat not in result:
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

def skip_n_levels(cat_map, cat, n):
    new_cat = cat
    while n > 0 and cat != None:
        new_cat = cat_map.get(new_cat, None)
        n -= 1
    return new_cat

cache = {}
def top_two_level(cat_map, cat):
    global cache

    if cat in cache:
        return (cat, cache[cat][0])

    seen = []
    super_cat = cat_map.get(cat, None)
    seen.append(cat)
    seen.append(super_cat)
    while cat_map.get(super_cat, None) != None:
        cat = super_cat
        super_cat = cat_map.get(super_cat, None)
        if super_cat in seen:
            print('Warning: detected loop %s, %s, %s' % (str(seen), cat, super_cat))
            break
        seen.append(super_cat)

    for i in range(len(seen) - 1):
        cache[seen[i]] = seen[i:]

    return (cat, super_cat)

def append_super_cat(dir, cat_map, article_map):
    cat_stat = Counter()
    supercat_stat = Counter()
    no_cat_count = 0
    no_supercat_count = 0

    cat_skip_levels = 4
    supercat_skip_levels = 2

    with open(dir + '/longabstract_en.nt', 'r') as f:
        # build stats
        for line in f:
            match = re.search(r'(^<http://dbpedia.org/[^<]+)\s', line)
            article_url = match.group(1)
            article_cat = skip_n_levels(cat_map, article_map.get(article_url, None), cat_skip_levels)
            if not article_cat:
                no_cat_count += 1
                continue
            article_super_cat = skip_n_levels(cat_map, article_cat, supercat_skip_levels)
            if not article_super_cat:
                no_supercat_count += 1
                continue
            cat_stat[article_cat] += 1
            supercat_stat[article_super_cat] += 1

    output_cat_stat = Counter()
    output_supercat_stat = Counter()
    with open(dir + '/longabstract_en.nt', 'r') as f:
        with open(dir + '/joinedlonabstract_en.nt', 'w') as g:
            for line in f:
                match = re.search(r'(^<http://dbpedia.org/[^<]+)\s', line)
                article_url = match.group(1)
                article_cat = skip_n_levels(cat_map, article_map.get(article_url, None), cat_skip_levels)
                if not article_cat:
                    continue
                article_super_cat = skip_n_levels(cat_map, article_cat, supercat_skip_levels)
                if not article_super_cat:
                    continue
                if cat_stat[article_cat] < 1000:
                    continue
                output_cat_stat[article_cat] += 1
                output_supercat_stat[article_super_cat] += 1
                g.write('%s %s %s\n' % (line.strip(), article_cat, article_super_cat))
                g.flush()

    print('Number of original categories: %d %d %d %d' % (no_cat_count, no_supercat_count, len(cat_stat), len(supercat_stat)))
    print('Number of output categories: %d %d %d %d' % (no_cat_count, no_supercat_count, len(output_cat_stat), len(output_supercat_stat)))

    with open(dir + '/catstats.txt', 'w') as f:
        f.write(str(dict(output_cat_stat)))

    with open(dir + '/supercatstats.txt', 'w') as f:
        f.write(str(dict(output_supercat_stat)))


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
