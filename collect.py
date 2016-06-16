"""collects web page html files from dmoz dataset."""
import sys
PY_VERS = (3, 2)
assert (sys.version_info >= PY_VERS), "Sorry, this program requires Python " \
    "{}+, not Python {}.".format('.'.join(map(str, PY_VERS)), sys.version[:5])

#########################################
# global import
#########################################
import os
import argparse
from collections import Counter
import xml.sax
import time
import json
import logging
# from GoogleScraper import scrape_with_config, GoogleSearchError
import concurrent.futures

#########################################
# FLAGS
#########################################
PARSER = argparse.ArgumentParser(description='collect web page html data.')
PARSER.add_argument("--data_dir",
                    type=str,
                    default="data/",
                    help="path of data directory.")
# CAT_FETCH is subset of CATEGORIES, only fetch those cats
PARSER.add_argument("--cat_fetch",
                    type=str,
                    default="Business,Society,Science,Recreation,Shopping,Games",
                    # default="Arts,Business,Computers,Health",
                    help="cats to be print, no space, empty to fetch all")
PARSER.add_argument("--pages_per_file",
                    type=int,
                    default=5000,
                    help="number of web pages per json file")
PARSER.add_argument("--max_file_num",
                    type=int,
                    default=2,
                    help="max num of files per cat, -1 for no limit")
PARSER.add_argument("--max_workers",
                    type=int,
                    default=16,
                    help="number of max threads to fetch web pages")
PARSER.add_argument("--fetch_timeout",
                    type=int,
                    default=15,
                    help="max seconds to fetch a web page")
PARSER.add_argument("--max_neighbor",
                    type=int,
                    default=15,
                    help="max # for every kinds of neighbors")
PARSER.add_argument("--max_child",
                    type=int,
                    default=20,
                    help="max # for children")
PARSER.add_argument("--max_html_length",
                    type=int,
                    default=512,
                    help="max length of each neighbors html length")
PARSER.add_argument("--dataset_type",
                    type=str,
                    default="ukwa",
                    help="dmoz or ukwa")
PARSER.add_argument("--cat_num",
                    type=int,
                    default=10,
                    help="number of categories to collect: 5 and 10")
FLAGS = PARSER.parse_args()

#########################################
# global variables
#########################################
CUR_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

if FLAGS.dataset_type == 'dmoz':
    if FLAGS.cat_num == 5:
        CATEGORIES = list(set(["Arts", "Business", "Computers", "Health",
                               "Sports"]))
    elif FLAGS.cat_num == 10:
        CATEGORIES = list(set(["Arts", "Business", "Computers", "Health",
                               'Society', 'Science', 'Sports', 'Recreation',
                               'Shopping', 'Games']))
        # CATEGORIES = list(set(["Arts", "Business", "Computers", "Health",
        #                        'Society', 'Science', 'Sports', 'Recreation',
        #                        'Shopping', 'Reference', 'Games', 'Home']))
    else:
        raise ValueError("cat_num wrong value: " + FLAGS.cat_num)
    DMOZ_JSON = "dmoz_{}.json".format(len(CATEGORIES))
elif FLAGS.dataset_type == 'ukwa':
    CATEGORIES = list(set(
        ['Arts & Humanities', 'Government, Law & Politics',
         'Society & Culture', 'Business, Economy & Industry',
         'Science & Technology', 'Medicine & Health', 'Education & Research',
         'Company Web Sites', 'Digital Society', 'Sports and Recreation']))
    DMOZ_JSON = "ukwa_{}.json".format(len(CATEGORIES))
else:
    raise ValueError("dataset_type has wrong value: {}".format(
        FLAGS.dataset_type))

DMOZ_FILE = "content.rdf.u8.gz"
DMOZ_URL = "http://rdf.dmoz.org/rdf/content.rdf.u8.gz"
UKWA_FILE = 'classification.tsv'

CAT_PREFIX = "Top/"

#########################################
# functions
#########################################


def maybe_download(data_dir, s_name, s_url):
    """download and unpack the source file if not exists.

    args:
        data_dir (str): paht of directory for storing data.
        s_name (str): name of compressed source file.
        s_url (str): where to download source file.

    return:
        str: path of unpacked source file directory.
    """
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        logging.info("created data_dir: {}".format(data_dir))

    s_packed = os.path.join(data_dir, s_name)
    logging.info("source path: " + s_packed)
    # split twice for .tar.**
    s_dir, s_ext = os.path.splitext(s_packed)
    if s_dir.endswith(".tar"):
        s_dir, e = os.path.splitext(s_dir)
        s_ext = e + s_ext

    # always create a new directory for unpacked files
    if not os.path.isdir(s_dir):
        os.mkdir(s_dir)
        logging.info("created source dir: " + s_dir)

    if os.listdir(s_dir):
        logging.info("file already exists:" + s_dir)
    else:
        if not os.path.isfile(s_packed):
            logging.info("downloading" + s_name + "...")
            import urllib.request
            import shutil
            # download_path should == s_packed
            # download_path, _ = urllib.urlretrieve(s_url, s_packed)
            with urllib.request.urlopen(s_url) as r, open(s_packed, 'wb') as f:
                shutil.copyfileobj(r, f)
            logging.info('Successfully downloaded' + s_packed)
            logging.info("size: {} bytes.".format(os.path.getsize(s_packed)))

        # uppack downloaded source file
        logging.info("extracting file:" + s_packed)
        if s_ext == ".tar.gz":
            import tarfile
            with tarfile.open(s_packed, "r:*") as f:
                f.extractall(s_dir)
        elif s_ext == ".bz2":
            # only single file!! need file name
            s = os.path.join(s_dir, os.path.basename(s_dir))
            import bz2
            data = bz2.BZ2File(s_packed).read()
            with open(s, "w") as s_unpack:
                s_unpack.write(data)
        elif s_ext == ".zip":
            import zipfile
            with zipfile.ZipFile(s_packed, "r") as z:
                z.extractall(s_dir)
        elif s_ext == ".gz":
            # only single file!! need file name
            s = os.path.join(s_dir, os.path.basename(s_dir))
            import gzip
            with gzip.open(s_packed, "rb") as f, open(s, "wb") as s_unpack:
                s_unpack.write(f.read())
        elif s_ext == "":
            logging.info("no file extention")
        else:
            raise ValueError("unknown compressed file")
        logging.info("successfully extracted file:")

    return s_dir


def read_json(filename):
    if os.path.isfile(filename):
        logging.info("reading from json file:" + filename)
        with open(filename) as data_file:
            data = json.load(data_file)
        logging.info("finish reading json file")
        return data
    else:
        raise FileNotFoundError("json file:", filename)


def write_json(filename, data):
    logging.info("writing dmoz to" + filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    logging.info("finish writing to" + filename)


class DmozHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self._current_page = ''
        self._capture_content = False
        self._current_content = {}
        self._expect_end = False
        self.count = []
        self.dmoz = {}
        for cat in CATEGORIES:
            self.dmoz.setdefault(cat, [])

    def startElement(self, name, attrs):
        if name == 'ExternalPage':
            self._current_page = attrs['about']
            self._current_content = {}
        elif name in ['d:Title', 'd:Description', 'priority', 'topic']:
            self._capture_content = True
            self._capture_content_type = name

    def endElement(self, name):
        # Make sure that the only thing after "topic" is "/ExternalPage"
        if self._expect_end:
            assert name == 'topic' or name == 'ExternalPage'
            if name == 'ExternalPage':
                self._expect_end = False

    def characters(self, content):
        if self._capture_content:
            assert not self._expect_end
            self._current_content[self._capture_content_type] = content
            # logging.info self._capture_content_type, self._current_content[self._capture_content_type]

            # This makes the assumption that "topic" is the last entity in each dmoz page:
            if self._capture_content_type == "topic":
                # add url to self.dmoz
                if content and content.startswith(CAT_PREFIX):
                    top_cat = content.split('/')[1]
                    if top_cat:
                        self.count.append(top_cat)
                    cat = content[len(CAT_PREFIX):]
                    for i in range(len(CATEGORIES)):
                        if cat.startswith(CATEGORIES[i]):
                            m_id = len(self.dmoz[CATEGORIES[i]])
                            title = self._current_content.get("d:Title", "")
                            desc = self._current_content.get("d:Description",
                                                             "")
                            self.dmoz[CATEGORIES[i]].append(
                                {"id": m_id,
                                 "url": self._current_page,
                                 "title": title,
                                 "desc": desc})
                            # logging.info("url:"+ self._current_page)
                self._expect_end = True
            self._capture_content = False


def parse_dmoz(file_path, dmoz_json):
    """parse dmoz xml content file.
    args:
        file_path (str): path of dmoz file.
    return:
        dmoz (dict): categories map to web pages.
            {category:[url_info{desc/id/url/title:value}]}
        count (Counter): # of url in every categories.
    """
    import random
    logging.info("parsing dmoz xml file... (it may take several minites)")
    # create an XMLReader
    parser = xml.sax.make_parser()
    handler = DmozHandler()
    parser.setContentHandler(handler)
    parser.parse(file_path)

    logging.info("dmoz:{}".format({k: len(handler.dmoz[k])
                                   for k in handler.dmoz.keys()}))
    logging.info("count:{}".format(Counter(handler.count)))
    logging.info("parsed dmoz xml successfully")
    logging.info("shuffling dmoz")
    for key in handler.dmoz:
        logging.info("shuffling category {}".format(key))
        random.shuffle(handler.dmoz[key])
        logging.info("reseting id")
        for i in range(len(handler.dmoz[key])):
            handler.dmoz[key][i]['id'] = i

    write_json(dmoz_json, handler.dmoz)


def parse_ukwa(ukwa_tsv, ukwa_json):
    """parse dmoz xml content file.
    args:
        ukwa_tsv (str): path of tsv file.
    return:
        dmoz (dict): categories map to web pages.
            {category:[url_info{desc/id/url/title:value}]}
        count (Counter): # of url in every categories.
    """
    import random
    import csv

    logging.info("parsing ukwa tsv file... (it may take several minites)")
    with open(ukwa_tsv) as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        ukwa = {cat: [] for cat in CATEGORIES}
        p_cats = []
        for row in tsvin:
            p_cats.append(row[0])
            if row[0] in CATEGORIES:
                ukwa[row[0]].append({"id": 0,
                                     "url": row[3],
                                     "title": row[2],
                                     "desc": ''})

    cnt = Counter(p_cats)
    logging.info("count:{}".format(Counter(cnt)))
    logging.info("parsed ukwa xml successfully")
    logging.info("shuffling ukwa")
    for key in ukwa:
        logging.info("shuffling category {}".format(key))
        random.shuffle(ukwa[key])
        logging.info("reseting id")
        for i in range(len(ukwa[key])):
            ukwa[key][i]['id'] = i

    write_json(ukwa_json, ukwa)


def chunks_generator(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def page_generator(aa):
    """Yield successive n-sized chunks from l."""
    for a in aa:
        yield a


def get_child_urls(main_page, max_child=20):
    """retrieve urls from giving html page.
    args:
        main_page(str): html file.
        max_child(int): max number of return urls.
    return:
        list of url string.
    """
    from bs4 import BeautifulSoup, SoupStrainer
    children = []
    for link in BeautifulSoup(main_page,
                              "html.parser",
                              parse_only=SoupStrainer('a')):
        if link.has_attr('href') and link['href'].startswith("http"):
            children.append(link['href'])
    if len(children) > max_child:
        children = children[:max_child]
    return children


def google_url(op, url, max_num=10):
    """get parent or spousal of url by google sear engine.
    args:
        op (str): "link" for parent urls or "related" for spousal urls
        url (str)
        max_num (int): number of return urls
    """
    s = op + ':' + url

    # See in the config.cfg file for possible values
    config = {
        'use_own_ip': True,
        'keyword': s,
        'search_engines': ['google'],
        'num_pages_for_keyword': round(max_num / 10),
        'scrape_method': 'selenium',
        'sel_browser': 'chrome',
        'do_caching': False,
        'log_level': 50
    }

    try:
        search = scrape_with_config(config)
        # let's inspect what we got
        urls = []
        for serp in search.serps:
            for link in serp.links:
                urls.append(link.link)
    except GoogleSearchError as e:
        logging.error("google_url: {}".format(e))
        raise
    return urls


def google_urls(urls, max_num=10):
    """get parent or spousal of url by google sear engine.
    args:
        op (str): "link" for parent urls or "related" for spousal urls
        url ([url_str, url_str])
        max_num (int): number of return urls
    """
    # See in the config.cfg file for possible values
    config = {
        'use_own_ip': True,
        'keywords': urls,
        'search_engines': ['google'],
        'num_pages_for_keyword': round(max_num / 10),
        'scrape_method': 'selenium',
        'sel_browser': 'chrome',
        'do_caching': False,
        'log_level': 50,
        'num_workers': 8
    }

    # config = {
    #     'use_own_ip': 'True',
    #     'keywords': urls,
    #     'search_engines': ['google'],
    #     'num_pages_for_keyword': round(max_num / 10),
    #     'scrape_method': 'http',
    #     'do_caching': 'False',
    #     'num_workers': 8
    # }
    try:
        logging.debug("a")
        search = scrape_with_config(config)
        # let's inspect what we got
        urls = []
        for serp in search.serps:
            similar = []
            for link in serp.links:
                similar.append(link.link)
            urls.append(similar)
    except GoogleSearchError as e:
        logging.error("google_url: {}".format(e))
        raise
    return urls


def create_nodes(pages):
    """
    args:
        [[url, html_text]]
    """
    nodes = []
    for page in pages:
        node = {}
        node["r_id"] = create_nodes.count
        node["r_url"] = page[0]
        node["r_html"] = page[1]
        create_nodes.count += 1
        nodes.append(node)
    return nodes


def get_htmls(urls, timeout):
    """get htmls from urls array.
    args:
        urls([str]): urls string array.
        timeout(int): seconds
    return:
        [url, html]
        url: str
        html: str
    """
    import urllib.request
    ret = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as conn:
                html = conn.read().decode("utf-8", errors="ignore")
        except:
            pass
        else:
            ret.append([url, html])
    return ret


def parse_htmls(htmls, max_len=-1, dmoz=None):
    """
    args:
        [[url, html]]
    return:
        [[url, text from html]]
    """
    from bs4 import BeautifulSoup
    import re
    new_htmls = []
    for html in htmls:
        soup = BeautifulSoup(html[1], "html.parser")
        new_html = []

        if dmoz and len(dmoz) == 2 and dmoz[0] and dmoz[1]:
            new_html.append('<core>' + dmoz[0] + '. ' + dmoz[1] + '</core>')

        if html[0]:
            new_html.append('<url>' + html[0] + '</url>')

        title = soup.title
        if title:
            new_html.append('<title>' + title.string.strip() + '</title>')

        meta = []
        for a in soup.find_all('meta'):
            if 'name' in a.attrs and (a['name'] == "description" or
                                      a['name'] == 'keywords'):
                meta.append(a['content'].strip())
        if meta:
            new_html.append('<meta>' + ' '.join(meta) + '</meata>')

        heading = []
        for a in soup.find_all(re.compile("h[1-5]")):
            heading.append(a.get_text().strip())
        if heading:
            new_html.append('<heading>' + ' '.join(heading) + '</heading>')

        paragraph = []
        for a in soup.find_all('p'):
            paragraph.append(a.get_text().strip())
        if paragraph:
            new_html.append('<paragraph>' + ' '.join(paragraph) +
                            '</paragraph>')

        tmp = ' '.join(new_html)
        if max_len > 0:
            if len(tmp) > max_len:
                tmp = tmp[:max_len]
        new_htmls.append([html[0], tmp])
    return new_htmls


def fetch_single_page(url, timeout):
    """Retrieve a single page and report the url and contents"""
    import urllib.request
    with urllib.request.urlopen(url["url"], timeout=timeout) as conn:
        main_page = conn.read().decode("utf-8", errors="ignore")

    page = {}
    page["id"] = url["id"]
    page["url"] = url["url"]
    main_pages = parse_htmls(
        [[page["url"], main_page]],
        FLAGS.max_html_length,
        dmoz=[url['title'], url['desc']])
    # page["nodes"].extend(create_nodes(main_pages))
    page['html'] = main_pages[0][1]
    assert isinstance(page['html'], str), "error: main_pages not string"
    logging.debug("add main")

    create_nodes.count = 0
    # page["edges"] = []
    page["relatives"] = []

    # get child neighbors
    try:
        child_urls = get_child_urls(main_page, max_child=FLAGS.max_child)
        child_pages = get_htmls(child_urls, timeout)
        child_pages = parse_htmls(child_pages, FLAGS.max_html_length)
        page["relatives"].extend(create_nodes(child_pages))
        logging.debug("add children")
    except Exception as e:
        logging.error("child: {}".format(e))

    # # get parent neighbors
    # parent_urls = google_url("link", url["url"], max_num=FLAGS.max_neighbor)
    # parent_pages = get_htmls(parent_urls, timeout)
    # page["relatives"].extend(create_nodes(parent_pages))

    # # get sibling neighbors
    # sibling_count = 0
    # for sibling in parent_pages:
    #     sibling_urls = get_child_urls(sibling[1], max_child=FLAGS.max_neighbor)
    #     sibling_pages = get_htmls(sibling_urls, timeout)
    #     page["relatives"].extend(create_nodes(sibling_pages))
    #     sibling_count += len(sibling_pages)

    # get similar neighbors
    # max_num should be 10 !!
    # try:
    #     if 'similar' in url and url['similar']:
    #         similar_pages = get_htmls(url['similar'], timeout)
    #         similar_pages = parse_htmls(similar_pages, 1024)
    #         page["relatives"].extend(create_nodes(similar_pages))
    #         logging.debug("add similar")
    # except Exception as e:
    #     logging.error("smilar page: {}".format(e))

    try:
        logging.debug("\nchild length: {}".format(len(child_pages)))
        # print("parent length: {}".format(len(parent_pages)))
        # print("sibling length: {}".format(sibling_count))
        # logging.debug("similar length: {}".format(len(similar_pages)))
        logging.info("url id: {}".format(url['id']))
    except Exception as e:
        logging.error("print: {}".format(e))

    return page


def fetch_chunk_pages(urls, max_workers):
    """
    args:
    return:
        number of valid pages downloaded in this chunk.
    """

    # try:
    #     # get similar pages for entire chunk
    #     logging.debug("get similar pages")
    #     tmp = ["related:" + u['url'] for u in urls]
    #     similar_urls = google_urls(tmp, max_num=10)
    #     for i in range(len(urls)):
    #         urls[i]['similar'] = similar_urls[i]
    #     logging.info("similar for chunk:{}".format(len(similar_urls)))
    # except Exception as e:
    #     logging.error("cannot google similar: {}".format(e))

    logging.info('start fetching chunk pages')
    pages = []
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(fetch_single_page, url, FLAGS.fetch_timeout): url
            for url in urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page = future.result()
            except Exception as exc:
                logging.error('%r:%r generated an exception: %s' %
                              (url["id"], url["url"], exc))
            else:
                # logging.info('%r page is %d bytes' % (url["url"], len(page)))
                pages.append(page)
    return pages


def fetch_pages(dmoz_json, out_dir):
    """grab html for every url in dmoz, then write to the json file.
    Both main and four kinds of relatives html will be collected.
    """
    dmoz = read_json(dmoz_json)
    keys = FLAGS.cat_fetch.split(',')
    if not keys[0]:
        keys = dmoz.keys()
    for key in keys:
        key_dir = os.path.join(out_dir, key)
        os.makedirs(key_dir)
        logging.info("\n\nwrite json files to folder: {}".format(key_dir))
        # urls = page_generator(dmoz[key])
        cur_chunk = []
        dmoz_head = 0
        j_ind = 0
        while j_ind < FLAGS.max_file_num and dmoz_head < len(dmoz[key]):
            t_start = time.time()
            sub = round((FLAGS.pages_per_file - len(cur_chunk)) * 1.5)
            while sub > 0 and len(
                    cur_chunk) < FLAGS.pages_per_file and dmoz_head < len(dmoz[
                        key]):
                logging.info("\n\nj_ind:{}, key:{}, sub:{}".format(j_ind, key,
                                                                   sub))
                dmoz_tail = dmoz_head + sub
                if dmoz_tail > len(dmoz[key]):
                    logging.info("reach the end of dmoz[key]")
                    dmoz_tail = len(dmoz[key])
                sub_chunk = fetch_chunk_pages(dmoz[key][dmoz_head:dmoz_tail],
                                              FLAGS.max_workers)
                cur_chunk.extend(sub_chunk)
                dmoz_head = dmoz_tail

            # write chunk json file
            chunk_file = os.path.join(key_dir, str(j_ind) + ".json")
            # write chunk of pages json file
            logging.info("\n\n\nfinish fetching chunk {}".format(j_ind))
            write_json(chunk_file, cur_chunk[:FLAGS.pages_per_file])
            cur_chunk = cur_chunk[FLAGS.pages_per_file:]
            duration = time.time() - t_start
            logging.info("total time used in chunk: {}s".format(duration))
            logging.info("average time of fetching one page: {}\n\n".format(
                duration / FLAGS.pages_per_file))
            j_ind += 1
        logging.info("\ntotal valid chunks # for {}: {}".format(key, j_ind))


def set_logging(stream=False, fileh=False, filename="example.log"):
    """set basic logging configurations (root logger).
    args:
        stream (bool): whether logging.info log to console.
        fileh (bool): whether write log to file.
        filename (str): the path of log file.
    return:
        configued root logger.
    """
    handlers = []
    level = logging.INFO
    log_format = '%(asctime)s: %(message)s'

    if stream:
        handlers.append(logging.StreamHandler())
    if fileh:
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(format=log_format, handlers=handlers, level=level)
    return logging.getLogger()


def main():
    print("start of main\n")

    html_dir = os.path.join(FLAGS.data_dir, "html_" + CUR_TIME)
    os.makedirs(html_dir)
    log_file = os.path.join(html_dir, "log")
    set_logging(stream=True, fileh=True, filename=log_file)
    logging.info("\nall arguments:")
    logging.info("CATEGORIES: {}".format(', '.join(CATEGORIES)))
    for arg in vars(FLAGS):
        logging.info("{:12}{}".format(arg, getattr(FLAGS, arg)))

    # download dmoz content file
    dmoz_path = maybe_download(FLAGS.data_dir, DMOZ_FILE, DMOZ_URL)
    dmoz_cont = os.path.join(dmoz_path, os.path.splitext(DMOZ_FILE)[0])
    if not os.path.isfile(dmoz_cont):
        raise FileNotFoundError("dmoz_cont")

    dmoz_json = os.path.join(FLAGS.data_dir, DMOZ_JSON)
    if os.path.isfile(dmoz_json):
        logging.info("file already exists, won't parse dmoz xml: {}".format(
            dmoz_json))
    else:
        if FLAGS.dataset_type == 'dmoz':
            # parse dmoz xml, and write to dmoz_json
            # json format: [url, url]
            # url: {"id":int, "url":str, "title":str, "desc":str}
            parse_dmoz(dmoz_cont, dmoz_json)
        else:
            ukwa_tsv = os.path.join(FLAGS.data_dir, UKWA_FILE)
            parse_ukwa(ukwa_tsv, dmoz_json)

    # grab main and relatives html text, write to disc
    # json format: [page, page]
    # page: {"id":int, "url":str, "edges":[edge, edge], "nodes":[node, node]}
    # the first node is the target page
    # edge: [n_id, n_id]
    # node: {"n_id":int, "n_url":str, "html":str, ???}

    # second edition:
    # json format: [page, page]
    # page: {"id":int, "url":str, "html":str, "relatives":[relative, relative]}
    # the first node is the target page
    # relative: {"r_id":int, "r_url":str, "r_html":str}
    fetch_pages(dmoz_json, html_dir)

    logging.info("\nend of main")


if __name__ == "__main__":
    main()
