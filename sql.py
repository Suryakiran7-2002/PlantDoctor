import sqlite3 as sq

conn = sq.connect('database.db')
cur = conn.cursor()

cur.execute('update info set youtube = (?) where key = "Tomato_Late_blight"',('<iframe width="999" height="638" src="https://www.youtube.com/embed/mHrcs1c_ToA" title="Late Blight Disease Management in Tomato & Potato Crops | BigHaat" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',))



conn.commit()
conn.close()