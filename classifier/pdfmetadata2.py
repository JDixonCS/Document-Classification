#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab ai si:

"""A simple GUI tool to edit PDF metadata (Subject, Author, Keywords etc.).

Requirements:
    pdftk (command line utility)
    PyGTK
    iterpipes
"""

# LICENSE: BSD-3
# Copyright: S. Astanin 2010

import sys
import re
import os
import locale
import shutil

from itertools import izip, chain
from functools import partial
from tempfile import mkstemp

import pygtk
pygtk.require('2.0')
import gtk

# http://pypi.python.org/pypi/iterpipes
from iterpipes import run, cmd, linecmd, strip, CalledProcessError

def partially_sorted(seq, head_items=[],key=None):
    """Put head_items in front and sort the rest as usual.

    >>> partially_sorted([])
    []
    >>> partially_sorted([2,1,3])
    [1, 2, 3]
    >>> partially_sorted([1,2,3], key = lambda x: -x)
    [3, 2, 1]
    >>> partially_sorted(range(6), head_items=[5,0], key = lambda x: -x)
    [5, 0, 4, 3, 2, 1]
    """
    heads = list(set(seq).intersection(set(head_items)))
    rest = list(set(seq) - set(head_items))
    heads.sort(key = lambda v: head_items.index(v))
    rest.sort(key = key)
    return heads + rest

# from http://wiki.python.org/moin/EscapingXml
import xml.parsers.expat as expat
def unescape(s):
    s = re.sub("\&#0;","",s)
    # the rest of this assumes that `s` is UTF-8
    list = []
    # create and initialize a parser object
    p = expat.ParserCreate("utf-8")
    p.buffer_text = True
    p.returns_unicode = True
    p.CharacterDataHandler = list.append
    # parse the data wrapped in a dummy element
    # (needed so the "document" is well-formed)
    p.Parse("<e>", 0)
    p.Parse(s, 0)
    p.Parse("</e>", 1)
    # join the extracted strings and return
    return u"".join(list)

class Gui:

    @staticmethod
    def iternumkeys(keyvalues):
        "Iterate through numbered key values (number, key)."
        # some fields should be displayed first
        keys = partially_sorted(list(keyvalues.iterkeys()),
                                ["Title", "Author", "Subject", "Keywords"])
        return izip(xrange(len(keyvalues)), keys)

    @staticmethod
    def insert_at(gtktable, object, row, column, padding=5):
        xmode = gtk.SHRINK if column == 0 else gtk.EXPAND | gtk.FILL
        gtktable.attach( object, column, column+1, row, row+1
                       , xpadding=padding, ypadding=padding
                       , xoptions=xmode)
        return gtktable

    def __init__(self, filename, fields):
        self.dialog = gtk.Dialog("Edit %s" % filename)
        self.dialog.set_size_request(480, 300)

        self.fields = fields.copy()
        self.fields_new = fields.copy()

        # add missing fields as empty
        for f in ["Author", "Title", "Subject", "Keywords"]:
            if not self.fields_new.has_key(f):
                self.fields_new.update({f: ""})

        tbl = gtk.Table(len(self.fields_new),4,False)
        self.dialog.vbox.add(tbl)

        # create input entries
        for i,n in self.iternumkeys(self.fields_new):
            l = gtk.Label(n)
            l.set_alignment(1,0.5)
            self.insert_at(tbl,l,i,0)

            v = self.fields_new[n]
            b = gtk.Entry()
            b.insert_text(v)
            b.connect("changed", self.update_fields, n)
            self.insert_at(tbl,b,i,1)

        self.dialog.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        self.dialog.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
        self.dialog.show_all()

    def update_fields(self, entry, name):
        self.fields_new.update({name: entry.get_text()})
        return self.fields_new

    def apply(self):
        self.fields = self.fields_new.copy()
        return self.fields

    def cancel(self):
        self.fields = {}
        return self.fields

    def run(self):
        resp = self.dialog.run()
        if resp == gtk.RESPONSE_OK:
            self.apply()
        else:
            self.cancel()
        return self.fields

def read_metadata(pdffile):
    """Read metadata from PDF file.
    Return (keyvalues, otherstrings), where
        keyvalues is a dictionary of key-value pairs
        otherstring are any other metadata strings which should be preserved.
    """
    cmdline = "pdftk \"%s\" dump_data output -" % pdffile
    meta = list(run(linecmd(cmdline) | strip()))
    kvregex = "^Info(Key|Value): "
    kvs = filter(lambda line: re.match(kvregex, line), meta)
    rest = list(set(meta) - set(kvs))
    kvs = map(lambda line: re.sub(kvregex, "", line), kvs)
    kvs = dict(zip(kvs[0::2], map(unescape, kvs[1::2])))
    return (kvs,rest)

def format_metadata(keyvalues, otherstrings):
    """Format metadata as in pdftk's dump_data. Inverse for read_metadata.

    >>> format_metadata({"spam":"eggs","a":"b"},["whatever"])
    'InfoKey: a\\nInfoValue: b\\nInfoKey: spam\\nInfoValue: eggs\\nwhatever\\n'
    """
    lines = map( lambda (k,v): ["InfoKey: %s" % k, "InfoValue: %s" % v]
               , keyvalues.iteritems() )
    lines = list(chain(*lines))
    return "\n".join(lines + otherstrings) + "\n"

def write_metadata(pdffile, metadata):
    """Update pdffile with new metadata."""
    # temporary pdf file
    fd, tmppdf = mkstemp(suffix=".pdf")
    os.close(fd) # only a temporary filename is needed
    # temporary file with metadata dump
    fd, tmpmeta = mkstemp(suffix=".txt")
    os.write(fd, metadata.encode(locale.getpreferredencoding()))
    os.close(fd)
    # run pdftk
    cmdline = u"pdftk \"%s\" update_info \"%s\" output \"%s\"" % \
               (pdffile, tmpmeta, tmppdf)
    err = os.system(cmdline)
    if not err: # success
        tmpsize = os.path.getsize(tmppdf)
        pdfsize = os.path.getsize(pdffile)
        if abs(pdfsize-tmpsize) < pdfsize/100:
            shutil.move(tmppdf, pdffile)
            os.remove(tmpmeta)
        else:
            if tmpsize == 0:
                raise RuntimeError("`%s' produced an empty file." % cmdline)
            else:
                raise RuntimeError("`%s' produced a file of wrong size."
                                   % cmdline)
    else:
        raise CalledProcessError(err, cmdline)

def process_file(f):
    kvs, other = read_metadata(f)
    new_kvs = Gui(f, kvs).run()
    if new_kvs:
        meta = format_metadata(new_kvs, other)
        write_metadata(f, meta)

def choose_file():
    d = gtk.FileChooserDialog("Select a file to edit")
    d.set_default_size(480,300)
    fpdf = gtk.FileFilter()
    fpdf.add_mime_type("application/pdf")
    fpdf.set_name("PDF")
    d.add_filter(fpdf)
    fany = gtk.FileFilter()
    fany.add_pattern("*")
    fany.set_name("All files")
    d.add_filter(fany)
    d.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
    d.add_button(gtk.STOCK_OPEN, gtk.RESPONSE_OK)
    r = d.run()
    if r == gtk.RESPONSE_OK:
        d.hide()
        return d.get_filename()
    else:
        return None


if __name__ == '__main__':
    if not sys.argv[1:]:
        f = choose_file()
        if f:
            process_file(f)
    else:
        for f in sys.argv[1:]:
            process_file(f)
