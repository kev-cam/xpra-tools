package XpraPod::Registry;
# Local registry of pods owned by this user.  Stand-in for a future
# NetMgr::Pod producer/client; the public API here (register/get/list/
# remove/migrate) is intentionally narrow so the swap-over is one file.
#
# Storage: ~/.config/xpra-pod/registry.json
#   { pods => { NAME => { name, host, ssh_port, user, flavor,
#                          parent_host, created, updated,
#                          history => [ { host, ssh_port, left }, ... ] } } }
#
# History records *previous* homes of a pod — appended when a register
# call comes in with a host different from what's already stored.  The
# current home is whatever's in the top-level fields.

use strict;
use warnings;
use JSON::PP qw(decode_json encode_json);
use File::Path qw(make_path);
use File::Basename qw(dirname);

sub _path {
    my $home = $ENV{HOME} or die "XpraPod::Registry: \$HOME unset\n";
    return "$home/.config/xpra-pod/registry.json";
}

sub _load {
    my $path = _path();
    return { pods => {} } unless -f $path;
    open my $fh, '<', $path or die "XpraPod::Registry: open $path: $!\n";
    local $/;
    my $raw = <$fh>;
    close $fh;
    return { pods => {} } unless defined $raw && length $raw;
    my $data = eval { decode_json($raw) };
    if (!ref $data || ref $data ne 'HASH') {
        warn "XpraPod::Registry: $path is not valid JSON; starting fresh\n";
        return { pods => {} };
    }
    $data->{pods} ||= {};
    return $data;
}

sub _save {
    my ($data) = @_;
    my $path = _path();
    my $dir  = dirname($path);
    make_path($dir, { mode => 0700 }) unless -d $dir;
    my $tmp = "$path.tmp.$$";
    open my $fh, '>', $tmp or die "XpraPod::Registry: open $tmp: $!\n";
    print $fh encode_json($data);
    close $fh or die "XpraPod::Registry: close $tmp: $!\n";
    chmod 0600, $tmp;
    rename $tmp, $path or die "XpraPod::Registry: rename $tmp -> $path: $!\n";
}

# register(name => ..., host => ..., ssh_port => ..., user => ...,
#          flavor => ..., parent_host => ...)
#
# Insert if new; update + push old home into history if the host changed.
# Returns the stored record.
sub register {
    my ($class, %pod) = @_;
    for my $k (qw(name host ssh_port user flavor)) {
        die "XpraPod::Registry::register: missing '$k'\n"
            unless defined $pod{$k} && length $pod{$k};
    }
    my $data = _load();
    my $now  = time;
    my $rec  = $data->{pods}{ $pod{name} };
    if ($rec) {
        # If the caller passed an explicit history, honor it (e.g.
        # pull-mode migrate is reconstructing history that was lost
        # by the intermediate fresh register).  Otherwise auto-append
        # the previous home on a host/port change.
        if (exists $pod{history}) {
            $rec->{history} = $pod{history};
        } elsif (($rec->{host} // '') ne $pod{host}
              || ($rec->{ssh_port} // 0) != $pod{ssh_port}) {
            push @{ $rec->{history} //= [] }, {
                host     => $rec->{host},
                ssh_port => $rec->{ssh_port},
                left     => $now,
            };
        }
        for my $k (qw(host ssh_port user flavor parent_host)) {
            $rec->{$k} = $pod{$k} if exists $pod{$k};
        }
        $rec->{updated} = $now;
    } else {
        $data->{pods}{ $pod{name} } = $rec = {
            history => [],
            %pod,                       # caller-provided history wins
            created => $now,
            updated => $now,
        };
    }
    _save($data);
    return $rec;
}

sub get {
    my ($class, $name) = @_;
    return _load()->{pods}{$name};
}

sub list {
    my ($class) = @_;
    return _load()->{pods};
}

sub remove {
    my ($class, $name) = @_;
    my $data = _load();
    my $rec  = delete $data->{pods}{$name};
    _save($data) if $rec;
    return $rec;
}

1;
