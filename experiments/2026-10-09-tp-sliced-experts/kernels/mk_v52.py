"""td_v52.cu = td_v51.cu with the entry in trace records (trace builds only):
176 n_hot = kind | ei << 4, n_cold = ready | q << 1; 141/142 n_hot = ei | q << 12;
172 n_hot = ei | q << 12."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v51.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''td_record_at(td_ss++, 176, td_c0, td_c1, td_now(), gr.kind, e.ready);''',
    '''td_record_at(td_ss++, 176, td_c0, td_c1, td_now(), gr.kind | ei << 4, e.ready | q << 1);''')
sub('''        td_record_at(td_fs++, 141 + done, td_hc, td_hd, td_now(), 0, 0);''',
    '''        td_record_at(td_fs++, 141 + done, td_hc, td_hd, td_now(), ei | q << 12, 0);''')
sub('''            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, 0, 0);''',
    '''            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, ei | q << 12, 0);''')
(here / "td_v52.cu").write_text(src)
print("ok")
