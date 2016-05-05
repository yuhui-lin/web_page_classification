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

#########################################
# FLAGS
#########################################
PARSER = argparse.ArgumentParser(description='collect web page html data.')
PARSER.add_argument("--data_dir",
                    type=str,
                    default="data/",
                    help="path of data directory.")
PARSER.add_argument("--pages_per_file",
                    type=int,
                    default=100,
                    help="number of web pages per json file")
PARSER.add_argument("--max_workers",
                    type=int,
                    default=8,
                    help="number of max threads to fetch web pages")
PARSER.add_argument("--fetch_timeout",
                    type=int,
                    default=10,
                    help="max seconds to fetch a web page")
FLAGS = PARSER.parse_args()

#########################################
# global variables
#########################################
CUR_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

DMOZ_FILE = "content.rdf.u8.gz"
DMOZ_URL = "http://rdf.dmoz.org/rdf/content.rdf.u8.gz"
DMOZ_JSON = "dmoz.json"

CATEGORIES = list(set(["Arts", "Business", "Computers", "Health"]))
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
        print("created data_dir:", data_dir)

    s_packed = os.path.join(data_dir, s_name)
    print("source path:", s_packed)
    # split twice for .tar.**
    s_dir, s_ext = os.path.splitext(s_packed)
    if s_dir.endswith(".tar"):
        s_dir, e = os.path.splitext(s_dir)
        s_ext = e + s_ext

    # always create a new directory for unpacked files
    if not os.path.isdir(s_dir):
        os.mkdir(s_dir)
        print("created source dir:", s_dir)

    if os.listdir(s_dir):
        print("file already exists:", s_dir)
    else:
        if not os.path.isfile(s_packed):
            print("downloading", s_name, "...")
            import urllib.request
            import shutil
            # download_path should == s_packed
            # download_path, _ = urllib.urlretrieve(s_url, s_packed)
            with urllib.request.urlopen(s_url) as r, open(s_packed, 'wb') as f:
                shutil.copyfileobj(r, f)
            print('Successfully downloaded', s_packed)
            print("size:", os.path.getsize(s_packed), 'bytes.')

        # uppack downloaded source file
        print("extracting file:", s_packed)
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
            print("no file extention")
        else:
            raise ValueError("unknown compressed file")
        print("successfully extracted file:")

    return s_dir


def read_json(filename):
    if os.path.isfile(filename):
        print("reading from json file:", filename)
        with open(filename) as data_file:
            data = json.load(data_file)
        print("finish reading json file")
        return data
    else:
        raise FileNotFoundError("json file:", filename)


def write_json(filename, data):
    print("writing dmoz to", filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
    print("finish writing to", filename)


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
            # print self._capture_content_type, self._current_content[self._capture_content_type]

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
                            # print("url:", self._current_page)
                self._expect_end = True
            self._capture_content = False


def parse_dmoz(file_path, dmoz_json):
    """parse dmoz xml content file.
    args:
        file_path (str): path of dmoz file.
    return:
        dmoz (dict): categories map to web pages.
        count (Counter): # of url in every categories.
    """
    print("parsing dmoz xml file... (it may take several minites)")
    # create an XMLReader
    parser = xml.sax.make_parser()
    handler = DmozHandler()
    parser.setContentHandler(handler)
    parser.parse(file_path)

    print("dmoz:", {k: len(handler.dmoz[k]) for k in handler.dmoz.keys()})
    print("count:", Counter(handler.count))
    print("parsed dmoz xml successfully")
    write_json(dmoz_json, handler.dmoz)


def chunks_generator(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def fetch_single_page(page, timeout):
    """Retrieve a single page and report the url and contents"""
    import urllib.request
    with urllib.request.urlopen(page["url"], timeout=timeout) as conn:
        return conn.read().decode("utf-8")


def fetch_chunk_pages(urls, outfile, max_workers):
    import concurrent.futures

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
                print('%r generated an exception: %s' % (url, exc))
            else:
                # print('%r page is %d bytes' % (url["url"], len(page)))
                pages.append(page)
    # write chunk of pages json file
    write_json(outfile, pages)


def fetch_pages(dmoz_json):
    """grab html for every url in dmoz, then write to the json file.
    Both main and four kinds of relatives html will be collected.
    """
    dmoz = read_json(dmoz_json)
    out_dir = os.path.join(FLAGS.data_dir, "html_" + CUR_TIME)
    os.makedirs(out_dir)
    for key in dmoz.keys():
        key_dir = os.path.join(out_dir, key)
        os.makedirs(key_dir)
        for index, urls in enumerate(chunks_generator(dmoz[key],
                                                      FLAGS.pages_per_file)):
            print("\nstart chunk:", str(index))
            t_start = time.time()
            chunk_file = os.path.join(key_dir, str(index) + ".json")
            fetch_chunk_pages(urls, chunk_file, FLAGS.max_workers)
            print("time used in chunk: {}s".format(time.time()-t_start))


def main():
    print("start of main\n")
    # download dmoz content file
    dmoz_path = maybe_download(FLAGS.data_dir, DMOZ_FILE, DMOZ_URL)
    dmoz_cont = os.path.join(dmoz_path, os.path.splitext(DMOZ_FILE)[0])
    if not os.path.isfile(dmoz_cont):
        raise FileNotFoundError("dmoz_cont")

    dmoz_json = os.path.join(FLAGS.data_dir, DMOZ_JSON)
    if os.path.isfile(dmoz_json):
        print("file already exists, won't parse dmoz xml:", dmoz_json)
    else:
        # parse dmoz xml, and write to dmoz_json
        # json format: [page, page]
        # page: {"id":int, "url":str, "title":str, "desc":str}
        parse_dmoz(dmoz_cont, dmoz_json)

    # grab main and relatives html text, write to disc
    # json format: [page, page]
    # page: {"id":int, "url":str, "edges":[edge, edge], "nodes":[node, node]}
    # edge: [n_id, n_id]
    # node: {"n_id":int, "n_url":str, "html":str, ???}
    fetch_pages(dmoz_json)

    print("\nend of main")


if __name__ == "__main__":
    main()
